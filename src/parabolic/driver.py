

import argparse
import json
import os
import sys

from parabolic.mdp import AlpacaMarketDataProvider, CachedMarketDataProvider


TIMEFRAME_MAP = {
    "minute": "1Min",
    "daily": "1Day",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch market data from Alpaca or TinyDB cache and print it to stdout.",
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. SPY")
    parser.add_argument("--start", required=True, help="Start datetime/date, e.g. 2024-01-01 or 2024-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="End datetime/date, e.g. 2024-01-31 or 2024-01-31T00:00:00Z")
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=sorted(TIMEFRAME_MAP.keys()),
        help="Bar timeframe to fetch",
    )
    parser.add_argument(
        "--adjustment",
        default="all",
        help="Alpaca adjustment mode, e.g. all, raw, split, dividend",
    )
    parser.add_argument("--feed", default=None, help="Optional Alpaca feed value")
    parser.add_argument(
        "--cache-dir",
        default=".cache/market_data",
        help="Directory containing the TinyDB cache file",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Alpaca API key. Falls back to ALPACA_API_KEY env var.",
    )
    parser.add_argument(
        "--api-secret",
        default=None,
        help="Alpaca API secret. Falls back to ALPACA_API_SECRET env var.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Fetch and print only the latest bar for the symbol",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ALPACA_API_KEY")
    api_secret = args.api_secret or os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print(
            "Missing Alpaca credentials. Set --api-key/--api-secret or ALPACA_API_KEY/ALPACA_API_SECRET.",
            file=sys.stderr,
        )
        return 1

    alpaca_provider = AlpacaMarketDataProvider(api_key=api_key, api_secret=api_secret)
    provider = CachedMarketDataProvider(
        cache_dir=args.cache_dir,
        alpaca_provider=alpaca_provider,
    )

    if args.latest:
        data = provider.get_latest_bar(symbol=args.symbol, feed=args.feed)
    else:
        data = provider.get_bars(
            symbol=args.symbol,
            timeframe=TIMEFRAME_MAP[args.timeframe],
            start=args.start,
            end=args.end,
            adjustment=args.adjustment,
            feed=args.feed,
        )

    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())