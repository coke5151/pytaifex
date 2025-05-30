# PyTaifex - Python Wrapper for Taiwan Futures Exchange TTB API

> **中文版**: [README.md](README.md)

PyTaifex is a Python wrapper library designed for the official TTB trading API of the Taiwan Futures Exchange (TAIFEX). It provides a concise, stable, and fully-featured interface, allowing developers to easily perform futures trading, market data subscription, position inquiry, and other operations.

## 🚀 Key Features

- **Real-time Market Data Subscription** - Supports simultaneous subscription to multiple products, receiving real-time quotes via callback functions.
- **Complete Order Management** - Supports the full lifecycle management of orders, including creation, price modification, quantity modification, and cancellation.
- **Position and Account Inquiry** - Provides real-time position inquiry and account margin information.
- **Multi-process Architecture** - Utilizes an independent process to handle TTB operations, ensuring the stability of the main program.
- **Comprehensive Error Handling** - Offers detailed exception types for easy error diagnosis and handling.
- **Context Manager Support** - Supports the `with` statement for automatic resource cleanup.
- **Complete Logging System** - Provides detailed operation logs for debugging and monitoring.

## 📋 System Requirements

- Python 3.13 or higher
- Official TAIFEX TTB API module file (TTBHelp.pyc)
- Official TAIFEX TTB software
- A TAIFEX trading competition account

## 🔧 Installation

### Using pip

```bash
pip install pytaifex
```

### Using uv

```bash
uv add pytaifex
```

### From Source

```bash
git clone [https://github.com/coke5151/pytaifex.git](https://github.com/coke5151/pytaifex.git)
cd pytaifex
pip install -e .
```

## 📖 Quick Start

### Basic Setup

First, you need to [download the TTB API module file and TTB software](https://sim2.taifex.com.tw/portal/tutorial) (TTBHelp.pyc) from the official TAIFEX website. Ensure that the TTB software is running, **you are logged into your account, and have selected the trading competition you wish to participate in**.

### Basic Usage Example

```python
from pytaifex import TTB, QuoteData, OrderSide, TimeInForce, OrderType

# Define quote callback function
def on_quote_received(quote_data: QuoteData):
    print(f"Quote received: {quote_data.symbol}")
    print(f"Latest price: {quote_data.price}")
    print(f"Bid prices: {quote_data.bid_ps}, Ask prices: {quote_data.ask_ps}")
    print(f"Tick time: {quote_data.tick_time}")

# Use Context Manager to ensure proper resource release (or call client.shutdown() manually)
with TTB("path/to/TTBHelp.pyc") as client:
    # Register quote callback function
    client.register_quote_callback(on_quote_received)

    # Subscribe to market data
    client.subscribe(["TXFF5", "MTXF5"])  # Subscribe to TAIFEX Futures and Mini TAIFEX Futures for June 2025

    # Create an order
    client.create_order(
        symbol1="TXFF5",           # Product code
        side1=OrderSide.BUY,       # Side: Buy
        price="17000",             # Order price
        time_in_force=TimeInForce.ROD,  # Time in force: Rest of Day
        order_type=OrderType.LIMIT,     # Order type: Limit order
        order_qty="1",             # Order quantity
        day_trade=False            # Is it day trading?
    )

    # Query orders
    orders = client.get_orders()
    for order in orders:
        print(f"Order number: {order.order_number}")
        print(f"Product: {order.symbol_name}")
        print(f"Status: {order.status}")

    # Query positions
    positions = client.get_positions()
    for position in positions:
        print(f"Position ID: {position.deal_id}")
        print(f"Product: {position.symbol1_name}")
        print(f"Unrealized P&L: {position.floating_profit_loss}")

    # Query account information
    accounts = client.get_accounts()
    for account in accounts:
        print(account)
```

## 📚 Detailed Usage Guide

### 1. Initialize TTB Client

```python
from pytaifex import TTB
import logging

# Create custom logger (optional)
logger = logging.getLogger("my_trading_app")
logger.setLevel(logging.INFO)

# Initialize TTB client
client = TTB(
    pyc_file_path="path/to/TTBHelp.pyc",  # Path to TTB API module
    host="http://localhost:8080",         # TTB server address (default)
    zmq_port=51141,                       # ZeroMQ port (default)
    logger=logger,                        # Custom logger (optional)
    timeout=5                             # Initialization timeout (seconds)
)
```

### 2. Market Data Subscription

```python
def quote_handler(quote: QuoteData):
    """Handle real-time quote data"""
    print(f"Product: {quote.symbol} ({quote.name})")
    print(f"Latest price: {quote.price}")
    print(f"Change: {quote.change_price} ({quote.change_ratio}%)")
    print(f"Bid price/volume: {quote.bid_ps}/{quote.bid_pv}")
    print(f"Ask price/volume: {quote.ask_ps}/{quote.ask_pv}")
    print(f"Volume: {quote.volume}")
    print("-" * 40)

# Register callback function
client.register_quote_callback(quote_handler)

# Subscribe to multiple products
symbols = ["TXFF5", "MTXF5", "TXO21000F5"]  # TAIFEX Futures, Mini TAIFEX Futures, TAIFEX Options
client.subscribe(symbols)
```

### 3. Order Management

#### Create Order

```python
# Limit Buy Order
client.create_order(
    symbol1="TXFF5",
    side1=OrderSide.BUY,
    price="21000",
    time_in_force=TimeInForce.ROD,  # ROD: Rest of Day, IOC: Immediate or Cancel, FOK: Fill or Kill
    order_type=OrderType.LIMIT,     # LIMIT: Limit order, MARKET: Market order
    order_qty="2",
    day_trade=True  # Day trading
)

# Spread Order (Inter-month Arbitrage)
client.create_order(
    symbol1="TXFF5",      # Near-month contract
    side1=OrderSide.BUY,
    symbol2="TXFG5",      # Far-month contract
    side2=OrderSide.SELL,
    price="50",           # Price difference (spread)
    time_in_force=TimeInForce.ROD,
    order_type=OrderType.LIMIT,
    order_qty="1",
    day_trade=False
)
```

#### Modify Order

```python
# Query existing orders
orders = client.get_orders()
if orders:
    order_number = orders[0].order_number

    # Modify price
    client.change_price(order_number, "20000")

    # Modify quantity
    client.change_qty(order_number, "3")

    # Cancel order
    client.cancel_order(order_number)
```

### 4. Position and Account Inquiry

```python
# Query positions
positions = client.get_positions()
for pos in positions:
    print(f"Position Information:")
    print(f"  Trade ID: {pos.deal_id}")
    print(f"  Main Product: {pos.symbol1_name} ({pos.symbol1_id})")
    print(f"  Side: {'Buy' if pos.side1 == OrderSide.BUY else 'Sell'}")
    print(f"  Holding Quantity: {pos.hold}")
    print(f"  Deal Price: {pos.deal_price}")
    print(f"  Settlement Price: {pos.settle_price}")
    print(f"  Unrealized P&L: {pos.floating_profit_loss}")
    print(f"  Currency: {pos.currency}")  # If it's a spread order

# Query account information
accounts = client.get_accounts()
for account in accounts:
    print(account) # account is a dict
```

### 5. Error Handling

```python
from pytaifex import (
    TTBConnectionError, TTBTimeoutError,
    OrderCreationError, OrderModificationError, OrderCancellationError,
    SubscribeError, ValidationError
)

try:
    client.create_order(
        symbol1="INVALID_SYMBOL",
        side1=OrderSide.BUY,
        price="17000",
        time_in_force=TimeInForce.ROD,
        order_type=OrderType.LIMIT,
        order_qty="1",
        day_trade=False
    )
except OrderCreationError as e:
    print(f"Order creation failed: {e}")
except TTBTimeoutError as e:
    print(f"Request timed out: {e}")
except TTBConnectionError as e:
    print(f"Connection error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔍 Product Code Description

The product code format for the Taiwan Futures Exchange is:

  - Futures: `Product Code + Month Code + Year Code`
  - Options: `Product Code + Strike Price + Month Code + Year Code`

### Month Code Table

  - A: January, B: February, C: March, D: April, E: May, F: June
  - G: July, H: August, I: September, J: October, K: November, L: December

### Common Product Examples

  - `TXFF5`: TAIFEX Futures June 2025 contract (TXF + F + 5)
  - `MTXF5`: Mini TAIFEX Futures June 2025 contract (MTX + F + 5)
  - `TXO21000F5`: TAIFEX Options June 2025, Strike Price 21000 contract (TXO + 21000 + F + 5)

## ⚠️ Important Notes

1.  **TTB Software Requirement**: Ensure the official TAIFEX TTB software is running before use.
2.  **API Module**: You need to download the latest TTBHelp.pyc file from the official website.
3.  **Network Connection**: Ensure a stable network connection to avoid trading interruptions.
4.  **Risk Management**: Use automated trading functions with caution. It is recommended to test in a simulated environment first.
5.  **Resource Management**: Call `client.shutdown()` after use or use a Context Manager (`with TTB(...) as client:`).

## 🐛 Troubleshooting

### Common Issues

**Q: Callback function not called after subscribing to market data.**
A: Please check:

  - Is the TTB software running?
  - Is the product code format correct?
  - Have you subscribed to the same product in the TTB software?
  - Is the product within trading hours?
  - Can you see quote updates in the TTB software?

**Q: Order creation failed.**
A: Please check:

  - Does the account have sufficient margin?
  - Is the product code correct?
  - Is the price within a reasonable range?
  - Is it within trading hours?

**Q: Connection timeout error.**
A: Please check:

  - TTB software connection status.
  - Is the network connection stable?
  - Are firewall settings blocking the connection?

## 📄 License

This project is licensed under the MIT License. For details, please refer to the [LICENSE](https://www.google.com/search?q=LICENSE) file.

## 🤝 Contribution Guidelines

Issues and Pull Requests are welcome\! Before submitting, please ensure:

1.  The code adheres to the project's coding style.
2.  New features include appropriate tests.
3.  Relevant documentation is updated.

## 📞 Contact Information

  - Author: pytree
  - Email: houjunqimail@gmail.com
  - GitHub: https://github.com/coke5151

-----

**Disclaimer**: This software is for learning and research purposes only. Users should bear their own trading risks, and the author is not responsible for any trading losses.