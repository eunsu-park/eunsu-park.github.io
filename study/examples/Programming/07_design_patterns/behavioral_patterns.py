"""
Behavioral Design Patterns

Patterns for communication between objects:
1. Observer - Define subscription mechanism to notify multiple objects
2. Strategy - Define family of algorithms, make them interchangeable
3. Command - Encapsulate request as object
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# =============================================================================
# 1. OBSERVER PATTERN
# Define one-to-many dependency so when one object changes state,
# all dependents are notified
# =============================================================================

print("=" * 70)
print("1. OBSERVER PATTERN")
print("=" * 70)


# Observer interface
class Observer(ABC):
    """Interface for observers that watch subjects"""

    @abstractmethod
    def update(self, subject: 'Subject') -> None:
        """Called when subject's state changes"""
        pass


# Subject (Observable)
class Subject:
    """
    Maintains list of observers and notifies them of state changes.
    """

    def __init__(self):
        self._observers: List[Observer] = []
        self._state: Any = None

    def attach(self, observer: Observer) -> None:
        """Add an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
            print(f"Attached observer: {observer.__class__.__name__}")

    def detach(self, observer: Observer) -> None:
        """Remove an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
            print(f"Detached observer: {observer.__class__.__name__}")

    def notify(self) -> None:
        """Notify all observers of state change"""
        for observer in self._observers:
            observer.update(self)

    @property
    def state(self) -> Any:
        return self._state

    @state.setter
    def state(self, value: Any) -> None:
        self._state = value
        self.notify()


# Concrete Subject - Stock
@dataclass
class Stock:
    """Stock price that observers watch"""
    symbol: str
    price: float


class StockMarket(Subject):
    """Observable stock market"""

    def __init__(self):
        super().__init__()
        self.stocks: Dict[str, Stock] = {}

    def update_stock(self, symbol: str, price: float) -> None:
        """Update stock price and notify observers"""
        old_price = self.stocks.get(symbol).price if symbol in self.stocks else 0
        self.stocks[symbol] = Stock(symbol, price)
        self._state = (symbol, old_price, price)
        self.notify()


# Concrete Observers
class EmailAlert(Observer):
    """Observer that sends email alerts"""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def update(self, subject: Subject) -> None:
        symbol, old_price, new_price = subject.state
        change = abs(new_price - old_price)

        if change >= self.threshold:
            print(f"[EMAIL] {symbol}: ${old_price:.2f} → ${new_price:.2f} "
                  f"(change: ${change:.2f})")


class SMSAlert(Observer):
    """Observer that sends SMS alerts"""

    def update(self, subject: Subject) -> None:
        symbol, old_price, new_price = subject.state
        if new_price > old_price:
            print(f"[SMS] {symbol} UP: ${new_price:.2f} (+{new_price - old_price:.2f})")
        elif new_price < old_price:
            print(f"[SMS] {symbol} DOWN: ${new_price:.2f} (-{old_price - new_price:.2f})")


class Logger(Observer):
    """Observer that logs all changes"""

    def update(self, subject: Subject) -> None:
        symbol, old_price, new_price = subject.state
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[LOG] {timestamp} | {symbol}: ${new_price:.2f}")


# =============================================================================
# 2. STRATEGY PATTERN
# Define family of algorithms, encapsulate each one, make them interchangeable
# =============================================================================

print("\n" + "=" * 70)
print("2. STRATEGY PATTERN")
print("=" * 70)


# Strategy interface
class PaymentStrategy(ABC):
    """Abstract payment strategy"""

    @abstractmethod
    def pay(self, amount: float) -> str:
        pass


# Concrete strategies
class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number: str, cvv: str):
        self.card_number = card_number
        self.cvv = cvv

    def pay(self, amount: float) -> str:
        # Mask card number
        masked = f"****-****-****-{self.card_number[-4:]}"
        return f"Paid ${amount:.2f} with Credit Card {masked}"


class PayPalPayment(PaymentStrategy):
    def __init__(self, email: str):
        self.email = email

    def pay(self, amount: float) -> str:
        return f"Paid ${amount:.2f} via PayPal ({self.email})"


class CryptoPayment(PaymentStrategy):
    def __init__(self, wallet_address: str, currency: str):
        self.wallet_address = wallet_address
        self.currency = currency

    def pay(self, amount: float) -> str:
        masked = f"{self.wallet_address[:6]}...{self.wallet_address[-4:]}"
        return f"Paid ${amount:.2f} in {self.currency} to {masked}"


class BankTransferPayment(PaymentStrategy):
    def __init__(self, account_number: str):
        self.account_number = account_number

    def pay(self, amount: float) -> str:
        masked = f"****{self.account_number[-4:]}"
        return f"Paid ${amount:.2f} via Bank Transfer to {masked}"


# Context that uses strategy
class ShoppingCart:
    """
    Shopping cart that can use different payment strategies.
    Strategy can be changed at runtime!
    """

    def __init__(self):
        self.items: List[tuple[str, float]] = []
        self._payment_strategy: Optional[PaymentStrategy] = None

    def add_item(self, name: str, price: float):
        self.items.append((name, price))

    def calculate_total(self) -> float:
        return sum(price for _, price in self.items)

    def set_payment_strategy(self, strategy: PaymentStrategy):
        """Change payment strategy at runtime"""
        self._payment_strategy = strategy

    def checkout(self) -> str:
        if not self._payment_strategy:
            return "Error: No payment method selected"

        total = self.calculate_total()
        return self._payment_strategy.pay(total)


# Another example: Sorting strategies
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass


class QuickSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class BubbleSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


class SortContext:
    """Context that uses sorting strategy"""

    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy

    def sort_data(self, data: List[int]) -> List[int]:
        return self._strategy.sort(data)


# =============================================================================
# 3. COMMAND PATTERN
# Encapsulate request as object, parameterize clients with different requests
# =============================================================================

print("\n" + "=" * 70)
print("3. COMMAND PATTERN")
print("=" * 70)


# Receiver - knows how to perform operations
class TextEditor:
    """Receiver that performs actual operations"""

    def __init__(self):
        self.text = ""

    def insert_text(self, text: str, position: int):
        self.text = self.text[:position] + text + self.text[position:]
        print(f"Inserted '{text}' at position {position}")

    def delete_text(self, start: int, length: int):
        deleted = self.text[start:start + length]
        self.text = self.text[:start] + self.text[start + length:]
        print(f"Deleted '{deleted}' from position {start}")
        return deleted

    def get_text(self) -> str:
        return self.text


# Command interface
class Command(ABC):
    """Abstract command"""

    @abstractmethod
    def execute(self) -> None:
        pass

    @abstractmethod
    def undo(self) -> None:
        pass


# Concrete commands
class InsertTextCommand(Command):
    """Command to insert text"""

    def __init__(self, editor: TextEditor, text: str, position: int):
        self.editor = editor
        self.text = text
        self.position = position

    def execute(self) -> None:
        self.editor.insert_text(self.text, self.position)

    def undo(self) -> None:
        self.editor.delete_text(self.position, len(self.text))


class DeleteTextCommand(Command):
    """Command to delete text"""

    def __init__(self, editor: TextEditor, start: int, length: int):
        self.editor = editor
        self.start = start
        self.length = length
        self.deleted_text = ""

    def execute(self) -> None:
        self.deleted_text = self.editor.delete_text(self.start, self.length)

    def undo(self) -> None:
        self.editor.insert_text(self.deleted_text, self.start)


# Invoker - manages command history
class CommandHistory:
    """Invoker that tracks command history for undo/redo"""

    def __init__(self):
        self.history: List[Command] = []
        self.redo_stack: List[Command] = []

    def execute_command(self, command: Command):
        """Execute command and add to history"""
        command.execute()
        self.history.append(command)
        self.redo_stack.clear()  # Clear redo stack on new command

    def undo(self):
        """Undo last command"""
        if not self.history:
            print("Nothing to undo")
            return

        command = self.history.pop()
        command.undo()
        self.redo_stack.append(command)
        print("Undone")

    def redo(self):
        """Redo last undone command"""
        if not self.redo_stack:
            print("Nothing to redo")
            return

        command = self.redo_stack.pop()
        command.execute()
        self.history.append(command)
        print("Redone")


# Another example: Smart home automation
class Light:
    """Receiver for light commands"""

    def __init__(self, location: str):
        self.location = location
        self.is_on = False
        self.brightness = 0

    def turn_on(self):
        self.is_on = True
        self.brightness = 100
        print(f"{self.location} light turned ON")

    def turn_off(self):
        self.is_on = False
        self.brightness = 0
        print(f"{self.location} light turned OFF")

    def set_brightness(self, level: int):
        if self.is_on:
            self.brightness = level
            print(f"{self.location} light brightness set to {level}%")


class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        self.light.turn_on()

    def undo(self):
        self.light.turn_off()


class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        self.light.turn_off()

    def undo(self):
        self.light.turn_on()


class RemoteControl:
    """Invoker - remote control with programmable buttons"""

    def __init__(self):
        self.commands: Dict[str, Command] = {}

    def set_command(self, button: str, command: Command):
        self.commands[button] = command

    def press_button(self, button: str):
        if button in self.commands:
            self.commands[button].execute()
        else:
            print(f"Button '{button}' not programmed")


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demonstrate_observer():
    print("\n[OBSERVER DEMONSTRATION]")
    print("-" * 50)

    # Create subject
    market = StockMarket()

    # Create observers
    email = EmailAlert(threshold=5.0)
    sms = SMSAlert()
    logger = Logger()

    # Attach observers
    market.attach(email)
    market.attach(sms)
    market.attach(logger)

    # Update stocks - all observers notified
    print("\nStock updates:")
    market.update_stock("AAPL", 150.00)
    market.update_stock("AAPL", 155.50)  # Big change
    market.update_stock("GOOGL", 2800.00)

    # Detach email observer
    print("\nDetaching email alerts...")
    market.detach(email)

    print("\nMore updates (no email):")
    market.update_stock("AAPL", 160.00)

    print("\nUse cases:")
    print("  • Event handling systems")
    print("  • MVC architecture")
    print("  • Real-time data feeds")
    print("  • Notification systems")


def demonstrate_strategy():
    print("\n[STRATEGY DEMONSTRATION]")
    print("-" * 50)

    # Create shopping cart
    cart = ShoppingCart()
    cart.add_item("Laptop", 1200.00)
    cart.add_item("Mouse", 25.00)
    cart.add_item("Keyboard", 75.00)

    print(f"Cart total: ${cart.calculate_total():.2f}\n")

    # Try different payment strategies
    print("Payment with Credit Card:")
    cart.set_payment_strategy(CreditCardPayment("1234567890123456", "123"))
    print(cart.checkout())

    print("\nPayment with PayPal:")
    cart.set_payment_strategy(PayPalPayment("user@example.com"))
    print(cart.checkout())

    print("\nPayment with Crypto:")
    cart.set_payment_strategy(CryptoPayment("0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb", "BTC"))
    print(cart.checkout())

    # Sorting example
    print("\n\nSorting Strategy:")
    data = [64, 34, 25, 12, 22, 11, 90]

    context = SortContext(QuickSort())
    print(f"QuickSort: {context.sort_data(data)}")

    context.set_strategy(BubbleSort())
    print(f"BubbleSort: {context.sort_data(data)}")

    print("\nUse cases:")
    print("  • Interchangeable algorithms")
    print("  • Payment processing")
    print("  • Data validation")
    print("  • Compression algorithms")


def demonstrate_command():
    print("\n[COMMAND DEMONSTRATION]")
    print("-" * 50)

    # Text editor with undo/redo
    editor = TextEditor()
    history = CommandHistory()

    print("Building text with commands:\n")

    # Execute commands
    history.execute_command(InsertTextCommand(editor, "Hello", 0))
    print(f"Text: '{editor.get_text()}'\n")

    history.execute_command(InsertTextCommand(editor, " World", 5))
    print(f"Text: '{editor.get_text()}'\n")

    history.execute_command(InsertTextCommand(editor, "!", 11))
    print(f"Text: '{editor.get_text()}'\n")

    # Undo operations
    print("Undoing operations:")
    history.undo()
    print(f"Text: '{editor.get_text()}'\n")

    history.undo()
    print(f"Text: '{editor.get_text()}'\n")

    # Redo
    print("Redoing:")
    history.redo()
    print(f"Text: '{editor.get_text()}'\n")

    # Smart home example
    print("\nSmart Home Remote Control:")
    living_room = Light("Living Room")
    bedroom = Light("Bedroom")

    remote = RemoteControl()
    remote.set_command("1", LightOnCommand(living_room))
    remote.set_command("2", LightOffCommand(living_room))
    remote.set_command("3", LightOnCommand(bedroom))
    remote.set_command("4", LightOffCommand(bedroom))

    print("\nPressing buttons:")
    remote.press_button("1")
    remote.press_button("3")
    remote.press_button("2")

    print("\nUse cases:")
    print("  • Undo/redo functionality")
    print("  • Transaction systems")
    print("  • Job queues")
    print("  • GUI buttons/menu items")
    print("  • Macro recording")


def print_summary():
    print("\n" + "=" * 70)
    print("BEHAVIORAL PATTERNS SUMMARY")
    print("=" * 70)

    print("""
OBSERVER
  Purpose: Notify multiple objects of state changes
  When: One-to-many relationships, event systems
  Benefits: Loose coupling, dynamic subscriptions
  Drawbacks: Can cause memory leaks, notification order issues

STRATEGY
  Purpose: Encapsulate algorithms, make them interchangeable
  When: Multiple algorithms for same task
  Benefits: Runtime algorithm selection, eliminates conditionals
  Drawbacks: Clients must know strategies, more objects

COMMAND
  Purpose: Encapsulate requests as objects
  When: Need undo/redo, queue operations, log requests
  Benefits: Decouples sender/receiver, supports undo
  Drawbacks: Many small classes, complexity

CHOOSING THE RIGHT PATTERN:
  • Observer: Multiple objects need updates
  • Strategy: Need to switch algorithms at runtime
  • Command: Need undo/redo or operation queuing

REAL-WORLD EXAMPLES:
  • Observer: Event listeners, data binding, pub-sub
  • Strategy: Payment methods, sorting, validation
  • Command: Text editors, transactions, smart home
""")


if __name__ == "__main__":
    demonstrate_observer()
    demonstrate_strategy()
    demonstrate_command()
    print_summary()
