"""
UML Class Diagram Generator (Text-Based)

Generates a formatted text representation of a UML class diagram.
Define classes with attributes and methods (using +/-/# visibility),
then specify relationships (inheritance, composition, association).

Example domain: e-commerce system with Product, Order, Customer, Payment.

Run:
    python 05_uml_class_diagram.py
"""

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Visibility = Literal["+", "-", "#"]
RelType = Literal["inheritance", "composition", "aggregation", "association", "dependency"]


@dataclass
class Attribute:
    name: str
    type_hint: str
    visibility: Visibility = "+"

    def render(self) -> str:
        return f"  {self.visibility} {self.name}: {self.type_hint}"


@dataclass
class Method:
    name: str
    params: str = ""
    return_type: str = "None"
    visibility: Visibility = "+"

    def render(self) -> str:
        return f"  {self.visibility} {self.name}({self.params}): {self.return_type}"


@dataclass
class UMLClass:
    name: str
    is_abstract: bool = False
    attributes: list[Attribute] = field(default_factory=list)
    methods: list[Method] = field(default_factory=list)

    def render(self) -> list[str]:
        lines: list[str] = []
        width = max(
            len(self.name) + 4,
            max((len(a.render()) for a in self.attributes), default=0) + 2,
            max((len(m.render()) for m in self.methods), default=0) + 2,
            20,
        )
        bar = "+" + "-" * width + "+"

        lines.append(bar)
        label = f"<<abstract>> {self.name}" if self.is_abstract else self.name
        lines.append(f"|{label.center(width)}|")
        lines.append(bar)

        if self.attributes:
            for attr in self.attributes:
                row = attr.render()
                lines.append(f"|{row:<{width}}|")
        else:
            lines.append(f"|{'(no attributes)':<{width}}|")

        lines.append(bar)

        if self.methods:
            for method in self.methods:
                row = method.render()
                lines.append(f"|{row:<{width}}|")
        else:
            lines.append(f"|{'(no methods)':<{width}}|")

        lines.append(bar)
        return lines


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: RelType
    label: str = ""
    source_mult: str = ""
    target_mult: str = ""

    _ARROWS: dict[str, str] = field(default_factory=lambda: {
        "inheritance":  "──────────▷",
        "composition":  "◆─────────",
        "aggregation":  "◇─────────",
        "association":  "──────────▶",
        "dependency":   "- - - - - ▶",
    })

    def render(self) -> str:
        arrow = self._ARROWS[self.rel_type]
        parts = [self.source]
        if self.source_mult:
            parts.append(f"[{self.source_mult}]")
        parts.append(arrow)
        if self.target_mult:
            parts.append(f"[{self.target_mult}]")
        parts.append(self.target)
        line = " ".join(parts)
        if self.label:
            line += f"  ({self.label})"
        return line


class UMLDiagram:
    def __init__(self, title: str):
        self.title = title
        self.classes: list[UMLClass] = []
        self.relationships: list[Relationship] = []

    def add_class(self, cls: UMLClass) -> "UMLDiagram":
        self.classes.append(cls)
        return self

    def add_relationship(self, rel: Relationship) -> "UMLDiagram":
        self.relationships.append(rel)
        return self

    def render(self) -> str:
        sections: list[str] = []

        header = f" UML Class Diagram: {self.title} "
        border = "=" * len(header)
        sections.append(border)
        sections.append(header)
        sections.append(border)
        sections.append("")

        # Classes
        sections.append("[ Classes ]")
        sections.append("")
        for cls in self.classes:
            sections.extend(cls.render())
            sections.append("")

        # Relationships
        if self.relationships:
            sections.append("[ Relationships ]")
            sections.append("")
            legend = {
                "inheritance":  "──────────▷  Inheritance (is-a)",
                "composition":  "◆─────────   Composition (owns, lifecycle tied)",
                "aggregation":  "◇─────────   Aggregation (has-a, independent lifecycle)",
                "association":  "──────────▶  Association (uses)",
                "dependency":   "- - - - - ▶  Dependency (depends on)",
            }
            seen_types: set[str] = set()
            for rel in self.relationships:
                sections.append(f"  {rel.render()}")
                seen_types.add(rel.rel_type)
            sections.append("")
            sections.append("[ Legend ]")
            for rtype, desc in legend.items():
                if rtype in seen_types:
                    sections.append(f"  {desc}")

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# E-commerce domain model
# ---------------------------------------------------------------------------

def build_ecommerce_diagram() -> UMLDiagram:
    diagram = UMLDiagram("E-Commerce System")

    # --- Product ---
    product = UMLClass(
        name="Product",
        attributes=[
            Attribute("id", "UUID", "-"),
            Attribute("name", "str", "-"),
            Attribute("price", "Decimal", "-"),
            Attribute("stock_qty", "int", "-"),
        ],
        methods=[
            Method("get_price", "", "Decimal"),
            Method("is_in_stock", "", "bool"),
            Method("apply_discount", "pct: float", "Decimal"),
        ],
    )

    # --- Customer ---
    customer = UMLClass(
        name="Customer",
        attributes=[
            Attribute("id", "UUID", "-"),
            Attribute("name", "str", "-"),
            Attribute("email", "str", "-"),
            Attribute("address", "Address", "#"),
        ],
        methods=[
            Method("place_order", "items: list[OrderLine]", "Order"),
            Method("get_order_history", "", "list[Order]"),
        ],
    )

    # --- OrderLine ---
    order_line = UMLClass(
        name="OrderLine",
        attributes=[
            Attribute("product", "Product", "+"),
            Attribute("quantity", "int", "+"),
            Attribute("unit_price", "Decimal", "+"),
        ],
        methods=[
            Method("subtotal", "", "Decimal"),
        ],
    )

    # --- Order ---
    order = UMLClass(
        name="Order",
        attributes=[
            Attribute("id", "UUID", "-"),
            Attribute("status", "OrderStatus", "-"),
            Attribute("created_at", "datetime", "-"),
            Attribute("lines", "list[OrderLine]", "-"),
        ],
        methods=[
            Method("total", "", "Decimal"),
            Method("add_line", "line: OrderLine", "None"),
            Method("cancel", "", "None"),
            Method("confirm", "", "None"),
        ],
    )

    # --- Payment (abstract) ---
    payment = UMLClass(
        name="Payment",
        is_abstract=True,
        attributes=[
            Attribute("amount", "Decimal", "#"),
            Attribute("currency", "str", "#"),
            Attribute("timestamp", "datetime", "#"),
        ],
        methods=[
            Method("process", "", "bool"),
            Method("refund", "", "bool"),
        ],
    )

    # --- CreditCardPayment ---
    cc_payment = UMLClass(
        name="CreditCardPayment",
        attributes=[
            Attribute("card_token", "str", "-"),
            Attribute("last_four", "str", "-"),
        ],
        methods=[
            Method("process", "", "bool"),
            Method("refund", "", "bool"),
        ],
    )

    for cls in [product, customer, order_line, order, payment, cc_payment]:
        diagram.add_class(cls)

    # Relationships
    diagram.add_relationship(Relationship("Customer", "Order",
        "association", "places", "1", "0..*"))
    diagram.add_relationship(Relationship("Order", "OrderLine",
        "composition", "contains", "1", "1..*"))
    diagram.add_relationship(Relationship("OrderLine", "Product",
        "association", "references", "1..*", "1"))
    diagram.add_relationship(Relationship("Order", "Payment",
        "association", "paid by", "1", "0..1"))
    diagram.add_relationship(Relationship("CreditCardPayment", "Payment",
        "inheritance"))

    return diagram


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    diagram = build_ecommerce_diagram()
    print(diagram.render())
