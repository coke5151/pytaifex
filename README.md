# PyTaifex - 臺灣期貨交易所 TTB API Python 封裝庫

> **English version available**: [README_en.md](https://github.com/coke5151/pytaifex/blob/main/README_en.md)

PyTaifex 是一個專為臺灣期貨交易所（TAIFEX）官方 TTB 交易 API 設計的 Python 封裝庫。它提供了一個簡潔、穩定且功能完整的介面，讓開發者能夠輕鬆地進行期貨交易、市場數據訂閱、部位查詢等操作。

## 🚀 主要特色

- **即時市場數據訂閱** - 支援多商品同時訂閱，透過回調函數接收即時報價
- **完整的委託管理** - 支援委託建立、修改價格、修改數量、取消委託等完整生命週期管理
- **部位與帳戶查詢** - 提供即時部位查詢和帳戶保證金資訊
- **多進程架構** - 採用獨立進程處理 TTB 操作，確保主程式穩定性
- **完善的錯誤處理** - 提供詳細的異常類型，便於錯誤診斷和處理
- **Context Manager 支援** - 支援 `with` 語句，自動管理資源清理
- **完整的日誌系統** - 提供詳細的操作日誌，便於除錯和監控

## 📋 系統需求

- Python 3.13 或更高版本
- 臺灣期貨交易所官方 TTB API 模組檔案 (TTBHelp.pyc)
- 臺灣期貨交易所官方 TTB 軟體
- 臺灣期貨交易所的交易競賽帳號

## 🔧 安裝方式

### 使用 pip 安裝

```bash
pip install pytaifex
```

### 使用 uv 安裝

```bash
uv add pytaifex
```

### 從原始碼安裝

```bash
git clone https://github.com/coke5151/pytaifex.git
cd pytaifex
pip install -e .
```

## 📖 快速開始

### 基本設定

首先，您需要從臺灣期貨交易所官方網站[下載 TTB API 模組檔案及 TTB 軟體](https://sim2.taifex.com.tw/portal/tutorial) (TTBHelp.pyc)，並確保 TTB 軟體正在運行並**已登入帳戶、選擇要交易的交易競賽**。

### 基本使用範例

```python
from pytaifex import TTB, QuoteData, OrderSide, TimeInForce, OrderType

# 定義報價 callback 函數
def on_quote_received(quote_data: QuoteData):
    print(f"收到報價: {quote_data.symbol}")
    print(f"最新價格: {quote_data.price}")
    print(f"買價: {quote_data.bid_ps}, 賣價: {quote_data.ask_ps}")
    print(f"時間: {quote_data.tick_time}")

# 使用 Context Manager 確保資源正確釋放（也可以手動呼叫 client.shutdown()）
with TTB("path/to/TTBHelp.pyc") as client:
    # 註冊報價回調函數
    client.register_quote_callback(on_quote_received)

    # 訂閱市場數據
    client.subscribe(["TXFF5", "MTXF5"])  # 訂閱 2025/06 的台指期和小台指期

    # 建立委託單
    client.create_order(
        symbol1="TXFF5",           # 商品代碼
        side1=OrderSide.BUY,       # 買賣別：買進
        price="17000",             # 委託價格
        time_in_force=TimeInForce.ROD,  # 委託時效：當日有效
        order_type=OrderType.LIMIT,     # 委託類型：限價單
        order_qty="1",             # 委託數量
        day_trade=False            # 是否為當沖
    )

    # 查詢委託單
    orders = client.get_orders()
    for order in orders:
        print(f"委託單號: {order.order_number}")
        print(f"商品: {order.symbol_name}")
        print(f"狀態: {order.status}")

    # 查詢部位
    positions = client.get_positions()
    for position in positions:
        print(f"部位ID: {position.deal_id}")
        print(f"商品: {position.symbol1_name}")
        print(f"未實現損益: {position.floating_profit_loss}")

    # 查詢帳戶資訊
    accounts = client.get_accounts()
    for account in accounts:
        print(account)
```

## 📚 詳細使用教學

### 1. 初始化 TTB 客戶端

```python
from pytaifex import TTB
import logging

# 建立自定義 logger（可選）
logger = logging.getLogger("my_trading_app")
logger.setLevel(logging.INFO)

# 初始化 TTB 客戶端
client = TTB(
    pyc_file_path="path/to/TTBHelp.pyc",  # TTB API 模組路徑
    host="http://localhost:8080",         # TTB 伺服器位址（預設）
    zmq_port=51141,                       # ZeroMQ 連接埠（預設）
    logger=logger,                        # 自定義 logger（可選）
    timeout=5                             # 初始化超時時間（秒）
)
```

### 2. 市場數據訂閱

```python
def quote_handler(quote: QuoteData):
    """處理即時報價數據"""
    print(f"商品: {quote.symbol} ({quote.name})")
    print(f"最新價: {quote.price}")
    print(f"漲跌: {quote.change_price} ({quote.change_ratio}%)")
    print(f"買價/量: {quote.bid_ps}/{quote.bid_pv}")
    print(f"賣價/量: {quote.ask_ps}/{quote.ask_pv}")
    print(f"成交量: {quote.volume}")
    print("-" * 40)

# 註冊 callback 函數
client.register_quote_callback(quote_handler)

# 訂閱多個商品
symbols = ["TXFF5", "MTXF5", "TXO21000F5"]  # 台指期、小台指、台指選擇權
client.subscribe(symbols)
```

### 3. 委託單管理

#### 建立委託單

```python
# 限價買單
client.create_order(
    symbol1="TXFF5",
    side1=OrderSide.BUY,
    price="21000",
    time_in_force=TimeInForce.ROD,  # ROD: 當日有效, IOC: 立即成交否則取消, FOK: 全部成交否則取消
    order_type=OrderType.LIMIT,     # LIMIT: 限價單, MARKET: 市價單
    order_qty="2",
    day_trade=True  # 當沖交易
)

# 價差單（跨月套利）
client.create_order(
    symbol1="TXFF5",      # 近月合約
    side1=OrderSide.BUY,
    symbol2="TXFG5",      # 遠月合約
    side2=OrderSide.SELL,
    price="50",           # 價差
    time_in_force=TimeInForce.ROD,
    order_type=OrderType.LIMIT,
    order_qty="1",
    day_trade=False
)
```

#### 修改委託單

```python
# 查詢現有委託
orders = client.get_orders()
if orders:
    order_number = orders[0].order_number

    # 修改價格
    client.change_price(order_number, "20000")

    # 修改數量
    client.change_qty(order_number, "3")

    # 取消委託
    client.cancel_order(order_number)
```

### 4. 部位與帳戶查詢

```python
# 查詢部位
positions = client.get_positions()
for pos in positions:
    print(f"部位資訊:")
    print(f"  交易ID: {pos.deal_id}")
    print(f"  主要商品: {pos.symbol1_name} ({pos.symbol1_id})")
    print(f"  買賣別: {'買進' if pos.side1 == OrderSide.BUY else '賣出'}")
    print(f"  持有數量: {pos.hold}")
    print(f"  成交價格: {pos.deal_price}")
    print(f"  結算價格: {pos.settle_price}")
    print(f"  未實現損益: {pos.floating_profit_loss}")
    print(f"  幣別: {pos.currency}")  # 如果是價差單

# 查詢帳戶資訊
accounts = client.get_accounts()
for account in accounts:
    print(account) # account 是一個 dict
```

### 5. 錯誤處理

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
    print(f"委託建立失敗: {e}")
except TTBTimeoutError as e:
    print(f"請求超時: {e}")
except TTBConnectionError as e:
    print(f"連線錯誤: {e}")
except Exception as e:
    print(f"未預期的錯誤: {e}")
```

## 🔍 商品代碼說明

臺灣期貨交易所的商品代碼格式為：
- 期貨：`商品代碼 + 月份代碼 + 年份代碼`
- 選擇權：`商品代碼 + 履約價 + 月份代碼 + 年份代碼`

### 月份代碼對照表
- A: 1月, B: 2月, C: 3月, D: 4月, E: 5月, F: 6月
- G: 7月, H: 8月, I: 9月, J: 10月, K: 11月, L: 12月

### 常見商品範例
- `TXFF5`: 台指期 2025 年 6 月合約 (TXF + F + 5)
- `MTXF5`: 小台指期 2025 年 6 月合約 (MTX + F + 5)
- `TXO21000F5`: 台指選擇權 2025 年 6 月，履約價 21000 的合約 (TXO + 21000 + F + 5)

## ⚠️ 重要注意事項

1. **TTB 軟體需求**: 使用前請確保臺灣期貨交易所官方 TTB 軟體正在運行
2. **API 模組**: 需要從官方網站下載最新的 TTBHelp.pyc 檔案
3. **網路連線**: 確保網路連線穩定，避免交易中斷
4. **風險管理**: 請謹慎使用自動交易功能，建議先在模擬環境測試
5. **資源管理**: 使用完畢後請呼叫 `client.shutdown()` 或使用 Context Manager (`with TTB(...) as client:`)

## 🐛 疑難排解

### 常見問題

**Q: 訂閱市場數據後 callback 函數沒有被呼叫**
A: 請檢查：
- TTB 軟體是否正在運行
- 商品代碼格式是否正確
- 確認在 TTB 軟體中同時訂閱了相同商品
- 該商品是否在交易時間內
- 你是否可以在 TTB 軟體裡看到有報價更新

**Q: 委託單建立失敗**
A: 請檢查：
- 帳戶是否有足夠保證金
- 商品代碼是否正確
- 價格是否在合理範圍內
- 是否在交易時間內

**Q: 連線超時錯誤**
A: 請檢查：
- TTB 軟體連線狀態
- 網路連線是否穩定
- 防火牆設定是否阻擋連線

## 📄 授權條款

本專案採用 MIT 授權條款。詳細內容請參閱 [LICENSE](LICENSE) 檔案。

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！在提交前請確保：

1. 程式碼符合專案的編碼風格
2. 新增功能包含適當的測試
3. 更新相關文件

## 📞 聯絡資訊

- 作者: pytree
- Email: houjunqimail@gmail.com
- GitHub: https://github.com/coke5151

---

**免責聲明**: 本軟體僅供學習和研究使用。使用者應自行承擔交易風險，作者不對任何交易損失負責。
