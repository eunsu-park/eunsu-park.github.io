"""
Creational Design Patterns

Patterns for object creation mechanisms:
1. Singleton - Ensure only one instance exists
2. Factory Method - Create objects without specifying exact class
3. Abstract Factory - Create families of related objects
4. Builder - Construct complex objects step by step
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from enum import Enum


# =============================================================================
# 1. SINGLETON PATTERN
# Ensure a class has only one instance and provide global access to it
# =============================================================================

print("=" * 70)
print("1. SINGLETON PATTERN")
print("=" * 70)


class SingletonMeta(type):
    """
    Metaclass for Singleton pattern.
    Thread-safe implementation.
    """
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    """
    Database connection that should exist only once.
    Uses Singleton to ensure single connection pool.
    """

    def __init__(self):
        self.connection_string = "postgresql://localhost:5432/mydb"
        self.pool_size = 10
        print(f"Initializing database connection to {self.connection_string}")

    def query(self, sql: str) -> str:
        return f"Executing: {sql}"


class ConfigManager(metaclass=SingletonMeta):
    """
    Application configuration manager.
    Should be singleton to ensure consistent config across app.
    """

    def __init__(self):
        self.settings: Dict[str, str] = {}
        print("Loading configuration...")

    def set(self, key: str, value: str):
        self.settings[key] = value

    def get(self, key: str) -> Optional[str]:
        return self.settings.get(key)


# =============================================================================
# 2. FACTORY METHOD PATTERN
# Define interface for creating objects, let subclasses decide which class
# =============================================================================

print("\n" + "=" * 70)
print("2. FACTORY METHOD PATTERN")
print("=" * 70)


# Product interface
class Document(ABC):
    """Abstract document that can be created"""

    @abstractmethod
    def open(self) -> str:
        pass

    @abstractmethod
    def save(self, content: str) -> str:
        pass


# Concrete products
class PDFDocument(Document):
    def open(self) -> str:
        return "Opening PDF document..."

    def save(self, content: str) -> str:
        return f"Saving to PDF: {content}"


class WordDocument(Document):
    def open(self) -> str:
        return "Opening Word document..."

    def save(self, content: str) -> str:
        return f"Saving to DOCX: {content}"


class TextDocument(Document):
    def open(self) -> str:
        return "Opening text document..."

    def save(self, content: str) -> str:
        return f"Saving to TXT: {content}"


# Creator (factory)
class DocumentCreator(ABC):
    """
    Abstract creator declares factory method.
    Subclasses override to change product type.
    """

    @abstractmethod
    def create_document(self) -> Document:
        """Factory method - subclasses implement this"""
        pass

    def new_document(self, content: str) -> str:
        """
        Business logic that uses the factory method.
        This code works with any document type!
        """
        doc = self.create_document()
        doc.open()
        return doc.save(content)


# Concrete creators
class PDFCreator(DocumentCreator):
    def create_document(self) -> Document:
        return PDFDocument()


class WordCreator(DocumentCreator):
    def create_document(self) -> Document:
        return WordDocument()


class TextCreator(DocumentCreator):
    def create_document(self) -> Document:
        return TextDocument()


# =============================================================================
# 3. ABSTRACT FACTORY PATTERN
# Create families of related objects without specifying concrete classes
# =============================================================================

print("\n" + "=" * 70)
print("3. ABSTRACT FACTORY PATTERN")
print("=" * 70)


# Abstract products
class Button(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


class Checkbox(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


# Concrete products - Windows family
class WindowsButton(Button):
    def render(self) -> str:
        return "Rendering Windows style button"


class WindowsCheckbox(Checkbox):
    def render(self) -> str:
        return "Rendering Windows style checkbox"


# Concrete products - Mac family
class MacButton(Button):
    def render(self) -> str:
        return "Rendering Mac style button"


class MacCheckbox(Checkbox):
    def render(self) -> str:
        return "Rendering Mac style checkbox"


# Concrete products - Linux family
class LinuxButton(Button):
    def render(self) -> str:
        return "Rendering Linux style button"


class LinuxCheckbox(Checkbox):
    def render(self) -> str:
        return "Rendering Linux style checkbox"


# Abstract factory
class GUIFactory(ABC):
    """Abstract factory for creating families of UI elements"""

    @abstractmethod
    def create_button(self) -> Button:
        pass

    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        pass


# Concrete factories
class WindowsFactory(GUIFactory):
    def create_button(self) -> Button:
        return WindowsButton()

    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()


class MacFactory(GUIFactory):
    def create_button(self) -> Button:
        return MacButton()

    def create_checkbox(self) -> Checkbox:
        return MacCheckbox()


class LinuxFactory(GUIFactory):
    def create_button(self) -> Button:
        return LinuxButton()

    def create_checkbox(self) -> Checkbox:
        return LinuxCheckbox()


# Client code
class Application:
    """
    Application works with factories and products through abstract interfaces.
    Doesn't know about concrete classes!
    """

    def __init__(self, factory: GUIFactory):
        self.factory = factory

    def create_ui(self) -> List[str]:
        button = self.factory.create_button()
        checkbox = self.factory.create_checkbox()
        return [button.render(), checkbox.render()]


# =============================================================================
# 4. BUILDER PATTERN
# Construct complex objects step by step
# =============================================================================

print("\n" + "=" * 70)
print("4. BUILDER PATTERN")
print("=" * 70)


class Computer:
    """Complex object with many optional parts"""

    def __init__(self):
        self.cpu: Optional[str] = None
        self.ram: Optional[int] = None
        self.storage: Optional[str] = None
        self.gpu: Optional[str] = None
        self.os: Optional[str] = None
        self.peripherals: List[str] = []

    def __str__(self) -> str:
        parts = [
            f"CPU: {self.cpu}",
            f"RAM: {self.ram}GB",
            f"Storage: {self.storage}",
        ]
        if self.gpu:
            parts.append(f"GPU: {self.gpu}")
        if self.os:
            parts.append(f"OS: {self.os}")
        if self.peripherals:
            parts.append(f"Peripherals: {', '.join(self.peripherals)}")
        return "Computer:\n  " + "\n  ".join(parts)


class ComputerBuilder:
    """
    Builder for constructing Computer step by step.
    Provides fluent interface.
    """

    def __init__(self):
        self.computer = Computer()

    def set_cpu(self, cpu: str) -> 'ComputerBuilder':
        self.computer.cpu = cpu
        return self

    def set_ram(self, ram: int) -> 'ComputerBuilder':
        self.computer.ram = ram
        return self

    def set_storage(self, storage: str) -> 'ComputerBuilder':
        self.computer.storage = storage
        return self

    def set_gpu(self, gpu: str) -> 'ComputerBuilder':
        self.computer.gpu = gpu
        return self

    def set_os(self, os: str) -> 'ComputerBuilder':
        self.computer.os = os
        return self

    def add_peripheral(self, peripheral: str) -> 'ComputerBuilder':
        self.computer.peripherals.append(peripheral)
        return self

    def build(self) -> Computer:
        """Return the constructed computer"""
        return self.computer


class ComputerType(Enum):
    """Predefined computer configurations"""
    GAMING = "gaming"
    OFFICE = "office"
    WORKSTATION = "workstation"


class ComputerDirector:
    """
    Director knows how to build specific configurations.
    Encapsulates common building recipes.
    """

    @staticmethod
    def build_gaming_pc() -> Computer:
        return (ComputerBuilder()
                .set_cpu("Intel i9-13900K")
                .set_ram(32)
                .set_storage("2TB NVMe SSD")
                .set_gpu("NVIDIA RTX 4090")
                .set_os("Windows 11")
                .add_peripheral("Gaming Keyboard")
                .add_peripheral("Gaming Mouse")
                .add_peripheral("144Hz Monitor")
                .build())

    @staticmethod
    def build_office_pc() -> Computer:
        return (ComputerBuilder()
                .set_cpu("Intel i5-12400")
                .set_ram(16)
                .set_storage("512GB SSD")
                .set_os("Windows 11 Pro")
                .add_peripheral("Wireless Keyboard")
                .add_peripheral("Wireless Mouse")
                .build())

    @staticmethod
    def build_workstation() -> Computer:
        return (ComputerBuilder()
                .set_cpu("AMD Threadripper PRO")
                .set_ram(128)
                .set_storage("4TB NVMe SSD")
                .set_gpu("NVIDIA RTX A6000")
                .set_os("Ubuntu 22.04 LTS")
                .add_peripheral("Mechanical Keyboard")
                .add_peripheral("Professional Mouse")
                .add_peripheral("4K Monitor")
                .add_peripheral("Graphics Tablet")
                .build())


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demonstrate_singleton():
    print("\n[SINGLETON DEMONSTRATION]")
    print("-" * 50)

    # Create two "instances" - should be the same object
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()

    print(f"db1 is db2: {db1 is db2}")  # True!
    print(f"db1 id: {id(db1)}, db2 id: {id(db2)}")

    # Configuration manager
    config1 = ConfigManager()
    config1.set("theme", "dark")

    config2 = ConfigManager()  # Same instance!
    print(f"Config theme: {config2.get('theme')}")  # Gets "dark"

    print("\nUse cases:")
    print("  • Database connection pools")
    print("  • Configuration managers")
    print("  • Logging services")
    print("  • Cache managers")


def demonstrate_factory_method():
    print("\n[FACTORY METHOD DEMONSTRATION]")
    print("-" * 50)

    # Client code works with abstract creator
    creators = [PDFCreator(), WordCreator(), TextCreator()]

    for creator in creators:
        result = creator.new_document("Hello, World!")
        print(result)

    print("\nUse cases:")
    print("  • Framework extension points")
    print("  • Plugin systems")
    print("  • When exact type isn't known until runtime")
    print("  • Delegating instantiation to subclasses")


def demonstrate_abstract_factory():
    print("\n[ABSTRACT FACTORY DEMONSTRATION]")
    print("-" * 50)

    # Determine platform (simulated)
    import platform
    os_name = platform.system()

    # Select appropriate factory
    if os_name == "Windows":
        factory = WindowsFactory()
    elif os_name == "Darwin":  # macOS
        factory = MacFactory()
    else:
        factory = LinuxFactory()

    # Application uses factory without knowing concrete types
    app = Application(factory)
    ui_elements = app.create_ui()

    print(f"Platform: {os_name}")
    for element in ui_elements:
        print(f"  {element}")

    print("\nUse cases:")
    print("  • Cross-platform UI toolkits")
    print("  • Database drivers (SQL/NoSQL families)")
    print("  • Creating themed UI elements")
    print("  • Multiple product families")


def demonstrate_builder():
    print("\n[BUILDER DEMONSTRATION]")
    print("-" * 50)

    # Build custom computer
    custom = (ComputerBuilder()
              .set_cpu("AMD Ryzen 9 7950X")
              .set_ram(64)
              .set_storage("1TB NVMe SSD")
              .set_gpu("AMD RX 7900 XTX")
              .set_os("Arch Linux")
              .add_peripheral("Mechanical Keyboard")
              .build())

    print("Custom Build:")
    print(custom)

    # Use director for predefined configs
    print("\nGaming PC:")
    gaming = ComputerDirector.build_gaming_pc()
    print(gaming)

    print("\nOffice PC:")
    office = ComputerDirector.build_office_pc()
    print(office)

    print("\nUse cases:")
    print("  • Complex object construction")
    print("  • Many optional parameters")
    print("  • Step-by-step construction")
    print("  • Different representations of same object")


def print_summary():
    print("\n" + "=" * 70)
    print("CREATIONAL PATTERNS SUMMARY")
    print("=" * 70)

    print("""
SINGLETON
  Purpose: Ensure only one instance exists
  When: Global state, shared resources
  Benefits: Controlled access, lazy initialization
  Drawbacks: Global state, hard to test

FACTORY METHOD
  Purpose: Create objects without specifying exact class
  When: Don't know exact types beforehand
  Benefits: Loose coupling, extensibility
  Drawbacks: More classes needed

ABSTRACT FACTORY
  Purpose: Create families of related objects
  When: Multiple product families
  Benefits: Consistency, isolation from concrete classes
  Drawbacks: Difficult to add new products

BUILDER
  Purpose: Construct complex objects step by step
  When: Many construction steps, optional parts
  Benefits: Fluent API, flexible construction
  Drawbacks: More code, complexity

CHOOSING THE RIGHT PATTERN:
  • Singleton: Need exactly ONE instance
  • Factory Method: Don't know exact product type
  • Abstract Factory: Need families of related products
  • Builder: Complex construction with many options
""")


if __name__ == "__main__":
    demonstrate_singleton()
    demonstrate_factory_method()
    demonstrate_abstract_factory()
    demonstrate_builder()
    print_summary()
