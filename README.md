# Tradeovate Algorithmic Trading Bot

## Overview

This is a production-ready algorithmic trading bot that connects directly to the Tradeovate API for futures trading. The bot implements multiple trading strategies with dynamic risk management, real-time market data analysis, and comprehensive backtesting capabilities.

## Features

- **Multiple Trading Strategies**: Momentum, Mean Reversion, Breakout, and Machine Learning-based predictive models
- **Robust Backtesting**: Evaluate and select the best strategy based on performance metrics
- **Dynamic Risk Management**: Position sizing, stop-loss, take-profit, and trailing stops
- **Real-time Market Data**: Direct integration with Tradeovate API for live data
- **Simulation Mode**: Test strategies without executing live trades
- **Flexible Configuration**: Easy customization via JSON or YAML configuration files
- **Comprehensive Logging**: Detailed activity and performance tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- Tradeovate account with API access
- API credentials (username, password, app_id, cid, sec)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tradeovate-bot.git
   cd tradeovate-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a configuration file (see Configuration section below)

## Usage

### Basic Usage

1. Start the bot in simulation mode:
   ```
   python main.py --config config/simulation_config.yaml
   ```

2. Start the bot in live trading mode:
   ```
   python main.py --config config/live_config.yaml
   ```

### Command Line Arguments

- `--config`: Path to configuration file (required)
- `--log-level`: Logging level (default: INFO)
- `--log-file`: Path to log file (default: logs/trading_bot.log)
- `--version`: Show version information and exit
- `--help`: Show help message and exit

## Configuration

The bot can be configured using either JSON or YAML files. Example configuration files are provided in the `config` directory.

### Configuration Options

#### General Settings

- `mode`: Trading mode (`live` or `simulation`)
- `account_id`: Tradeovate account ID (required for live mode)
- `initial_capital`: Initial capital for backtesting and simulation
- `commission_rate`: Commission rate per trade (percentage)
- `slippage`: Slippage per trade (percentage)
- `status_interval_minutes`: Status logging interval in minutes

#### Simulation Settings

- `simulation_speed`: Speed multiplier (higher = faster)
- `simulation_data_dir`: Directory containing historical data files

#### API Configuration

- `api.username`: Tradeovate username
- `api.password`: Tradeovate password
- `api.app_id`: Application ID
- `api.app_version`: Application version
- `api.cid`: Client ID provided by Tradeovate
- `api.sec`: Secret key provided by Tradeovate
- `api.device_id`: Optional device identifier
- `api.use_live`: Whether to use live environment (true) or demo (false)
- `api.token_refresh_margin`: Seconds before token expiry to refresh
- `api.max_retries`: Maximum number of API call retries
- `api.retry_delay`: Delay between retries in seconds

#### Risk Management

- `risk.max_position_size`: Maximum position size in contracts
- `risk.max_positions`: Maximum number of open positions
- `risk.max_drawdown_pct`: Maximum drawdown percentage allowed
- `risk.max_daily_loss_pct`: Maximum daily loss percentage allowed
- `risk.max_trade_risk_pct`: Maximum risk percentage per trade
- `risk.position_size_method`: Position sizing method (`fixed`, `percent_risk`, or `kelly`)
- `risk.kelly_fraction`: Fraction of Kelly criterion to use (0.0 to 1.0)
- `risk.use_stop_loss`: Whether to use stop loss
- `risk.stop_loss_atr_multiple`: Stop loss as multiple of ATR
- `risk.fixed_stop_loss_pct`: Fixed stop loss percentage
- `risk.use_trailing_stop`: Whether to use trailing stop
- `risk.trailing_stop_atr_multiple`: Trailing stop as multiple of ATR
- `risk.fixed_trailing_stop_pct`: Fixed trailing stop percentage
- `risk.use_take_profit`: Whether to use take profit
- `risk.take_profit_atr_multiple`: Take profit as multiple of ATR
- `risk.fixed_take_profit_pct`: Fixed take profit percentage
- `risk.max_open_time_minutes`: Maximum time to keep a position open in minutes

#### Trading Strategies

Strategies are defined as a list under the `strategies` key. Each strategy has the following properties:

- `type`: Strategy type (`momentum`, `mean_reversion`, `breakout`, or `ml_predictive`)
- `symbol`: Trading symbol
- `name`: Strategy name (optional)
- `parameters`: Strategy-specific parameters

### Example Configuration

```yaml
# Trading mode: "live" or "simulation"
mode: "simulation"

# Account ID for live trading (required for live mode)
account_id: null

# Initial capital for backtesting and simulation
initial_capital: 100000.0

# Risk management settings
risk:
  max_position_size: 1
  max_positions: 3
  max_drawdown_pct: 5.0
  position_size_method: "fixed"
  use_stop_loss: true
  fixed_stop_loss_pct: 1.0

# Trading strategies
strategies:
  - type: "momentum"
    symbol: "ES"
    name: "ES Momentum"
    parameters:
      fast_period: 10
      slow_period: 30
      signal_threshold: 0.0005
```

## Trading Strategies

### Momentum Strategy

The momentum strategy identifies and trades in the direction of price trends. It uses two moving averages (fast and slow) to determine trend direction and generates signals when the fast moving average crosses above or below the slow moving average.

#### Parameters

- `fast_period`: Period for the fast moving average
- `slow_period`: Period for the slow moving average
- `signal_threshold`: Minimum difference between fast and slow MAs to generate a signal
- `stop_loss_pct`: Stop loss percentage
- `take_profit_pct`: Take profit percentage
- `use_trailing_stop`: Whether to use trailing stop
- `trailing_stop_pct`: Trailing stop percentage

### Mean Reversion Strategy

The mean reversion strategy identifies overbought and oversold conditions and trades on the assumption that prices will revert to their mean. It uses Bollinger Bands to identify extreme price movements and generates signals when prices move beyond a certain threshold.

#### Parameters

- `lookback_period`: Period for calculating the mean and standard deviation
- `std_dev_threshold`: Number of standard deviations for entry signals
- `exit_threshold`: Number of standard deviations for exit signals
- `stop_loss_pct`: Stop loss percentage
- `take_profit_pct`: Take profit percentage

### Breakout Strategy

The breakout strategy identifies and trades price breakouts from consolidation periods. It uses price channels to identify support and resistance levels and generates signals when prices break out of these levels.

#### Parameters

- `breakout_period`: Period for calculating price channels
- `confirmation_period`: Number of bars to confirm a breakout
- `volatility_filter`: Whether to use volatility filter
- `volatility_period`: Period for calculating volatility
- `volatility_threshold`: Volatility threshold for filtering signals
- `stop_loss_pct`: Stop loss percentage
- `take_profit_pct`: Take profit percentage

### Machine Learning Predictive Strategy

The ML-based predictive strategy uses a Random Forest classifier to predict price movements based on technical indicators. It generates signals based on the predicted probability of price increases or decreases.

#### Parameters

- `lookback_window`: Number of periods to use for feature creation
- `prediction_horizon`: Number of periods ahead to predict
- `training_ratio`: Ratio of data to use for training
- `retrain_interval`: Number of bars between model retraining
- `min_training_samples`: Minimum samples required for training
- `confidence_threshold`: Minimum prediction confidence for signal generation
- `stop_loss_pct`: Stop loss percentage
- `take_profit_pct`: Take profit percentage
- `use_trailing_stop`: Whether to use trailing stop
- `trailing_stop_pct`: Trailing stop percentage
- `model_path`: Path to save/load model (empty for no persistence)
- `feature_importance_threshold`: Minimum feature importance to keep feature

## Backtesting

The bot includes a comprehensive backtesting framework for evaluating and selecting trading strategies. Backtesting can be performed using historical data to simulate trading and measure performance metrics.

### Performance Metrics

- **Sharpe Ratio**: Risk-adjusted return
- **Profit Factor**: Ratio of gross profit to gross loss
- **Win Rate**: Percentage of winning trades
- **Maximum Drawdown**: Maximum peak-to-trough decline
- **Average Trade**: Average profit/loss per trade
- **Total Return**: Total percentage return
- **Annualized Return**: Annualized percentage return

## Security and Compliance

### API Key Storage

API keys and credentials are stored securely in the configuration file. For production use, it is recommended to:

1. Use environment variables for sensitive information
2. Restrict access to configuration files
3. Consider using a secure vault mechanism for credential storage

### Regulatory Compliance

The bot is designed to comply with exchange rules and regulations. However, users are responsible for ensuring that their trading activities comply with all applicable laws and regulations.

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API credentials in configuration file
   - Check account status and permissions

2. **Connection Issues**
   - Check internet connection
   - Verify Tradeovate API status
   - Check firewall settings

3. **Order Placement Failures**
   - Verify account has sufficient funds
   - Check for trading restrictions
   - Verify contract specifications

### Logging

Detailed logs are written to the specified log file. Increase the log level for more detailed information:

```
python main.py --config config/live_config.yaml --log-level DEBUG
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
