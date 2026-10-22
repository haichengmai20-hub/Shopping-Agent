"""商品目录与独立判定：哪些商品真正满足买家的条件。

训练环境里的检索调用线上真实工具；奖励需要一份不依赖策略行为的标准答案。这里复用线上的
商品数据、汇率表和报价规则，按与检索用例相同的口径（第一个有库存的规格、按数量 1 报到手价）
逐件判定。买家说的预算一律按到手价理解，这正是线上价格上限只比单件价、需要另外核对的地方。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.infrastructure.persistence.seed_products import build_seed_products

CURRENCY = "CNY"
MATERIAL_TAGS = ("合成聚合物", "金属", "天然纤维", "玻璃", "陶瓷", "纸", "天然材料")


@dataclass(frozen=True)
class Offer:
    """一件商品在判定时用到的全部事实。"""

    product_id: str
    title: str
    category: str
    unit_price: float  # 目标币种下的单件商品价，与线上商品卡 price_major 同口径
    landed: dict  # 目的地 → 数量 1 的到手价；不可配送的目的地不在表里
    materials: frozenset
    in_stock: bool


class Catalog:
    """只读的商品目录。任务只检索、不下单，所以每条轨迹都从同一份目录出发，不需要重置库存。"""

    def __init__(self, products=None):
        self.products = list(products) if products is not None else build_seed_products()
        self.tariff = TariffSchedule(rates=ExchangeRateTable())
        self.offers = {p.product_id: self._offer(p) for p in self.products}

    def _offer(self, product) -> Offer:
        sku = product.primary_available_sku()
        landed = {}
        if sku is not None:
            for dest in sorted(set(product.ships_to) & set(self.tariff.supported_destinations())):
                quote = self.tariff.quote(subtotal=sku.price, category=product.category, ship_to=dest,
                                          quantity=1, target_currency=CURRENCY).to_dict()
                landed[dest] = float(quote["landed_total_major"])
        unit = self.tariff.rates.convert(sku.price, CURRENCY).to_major_units() if sku is not None else None
        return Offer(product.product_id, product.title, product.category, unit, landed,
                     frozenset(product.material_tags), sku is not None)

    def violations(self, product_id: str, keyword: str, ship_to: str, budget: float, excluded) -> list[str]:
        """返回这件商品违反的条件；空列表表示满足。商品不存在时返回 ['unknown_product']。"""
        offer = self.offers.get(product_id)
        if offer is None:
            return ["unknown_product"]
        reasons = []
        if keyword not in offer.title:
            reasons.append("keyword")
        if not offer.in_stock:
            reasons.append("out_of_stock")
        if ship_to not in offer.landed:
            reasons.append("ship_to")
        if offer.materials & set(excluded):
            reasons.append("material")
        if ship_to in offer.landed and offer.landed[ship_to] > budget:
            reasons.append("over_budget")
        return reasons

    def feasible(self, keyword: str, ship_to: str, budget: float, excluded=()) -> list[str]:
        """满足全部条件的商品，按到手价从低到高排列。"""
        hits = [pid for pid in self.offers if not self.violations(pid, keyword, ship_to, budget, excluded)]
        return sorted(hits, key=lambda pid: (self.offers[pid].landed[ship_to], pid))

    def traps(self, keyword: str, ship_to: str, budget: float, excluded=()) -> list[str]:
        """陷阱商品：单件价不超过预算、能通过线上价格上限过滤，但到手价超预算。"""
        out = []
        for pid, offer in self.offers.items():
            if self.violations(pid, keyword, ship_to, budget, excluded) == ["over_budget"] and offer.unit_price <= budget:
                out.append(pid)
        return sorted(out)

    def eligible(self, keyword: str, ship_to: str, excluded=()) -> list[str]:
        """不看预算时满足其余条件的商品，用于挑选预算。"""
        return [pid for pid in self.offers if set(self.violations(pid, keyword, ship_to, 10 ** 9, excluded)) == set()]


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    return Catalog()
