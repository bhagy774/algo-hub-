"""Explicit network diagnostic; never run during offline test collection."""
import os

def main():
    if os.environ.get("JARVIS_RUN_NETWORK_DIAGNOSTICS") != "1":
        raise SystemExit("Set JARVIS_RUN_NETWORK_DIAGNOSTICS=1 and configure your own Upstox credentials.")
    import requests, json, sys
    sys.stdout.reconfigure(encoding="utf-8")

    TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    }

    print("=" * 50)
    print("   UPSTOX LIVE TOKEN TEST")
    print("=" * 50)

    # Test 1: User Profile
    print("\n[1] User Profile...")
    r = requests.get("https://api.upstox.com/v2/user/profile", headers=headers, timeout=10)
    print(f"    Status: {r.status_code}")
    if r.status_code == 200:
        d = r.json().get("data", {})
        print(f"    Name  : {d.get('user_name', 'N/A')}")
        print(f"    Email : {d.get('email', 'N/A')}")
        print(f"    Broker: {d.get('broker', 'N/A')}")
        print("    >>> TOKEN IS VALID! <<<")
    else:
        print(f"    Error: {r.text[:400]}")

    # Test 2: Nifty 50 LTP
    print("\n[2] Nifty 50 Live Price...")
    r2 = requests.get(
        "https://api.upstox.com/v2/market-quote/ltp",
        params={"symbol": "NSE_INDEX|Nifty 50"},
        headers=headers, timeout=10
    )
    print(f"    Status: {r2.status_code}")
    if r2.status_code == 200:
        data2 = r2.json()
        print(f"    Response: {json.dumps(data2, indent=4)}")
    else:
        print(f"    Error: {r2.text[:400]}")

    # Test 3: Bank Nifty LTP
    print("\n[3] Bank Nifty Live Price...")
    r3 = requests.get(
        "https://api.upstox.com/v2/market-quote/ltp",
        params={"symbol": "NSE_INDEX|Nifty Bank"},
        headers=headers, timeout=10
    )
    print(f"    Status: {r3.status_code}")
    if r3.status_code == 200:
        data3 = r3.json()
        print(f"    Response: {json.dumps(data3, indent=4)}")
    else:
        print(f"    Error: {r3.text[:400]}")

    # Test 4: RELIANCE equity
    print("\n[4] RELIANCE Stock Price...")
    r4 = requests.get(
        "https://api.upstox.com/v2/market-quote/ltp",
        params={"symbol": "NSE_EQ|INE002A01018"},
        headers=headers, timeout=10
    )
    print(f"    Status: {r4.status_code}")
    if r4.status_code == 200:
        data4 = r4.json()
        print(f"    Response: {json.dumps(data4, indent=4)}")
    else:
        print(f"    Error: {r4.text[:400]}")

    # Test 5: Funds/Margin
    print("\n[5] Account Funds/Margin...")
    r5 = requests.get("https://api.upstox.com/v2/user/get-funds-and-margin", headers=headers, timeout=10)
    print(f"    Status: {r5.status_code}")
    if r5.status_code == 200:
        data5 = r5.json()
        print(f"    Response: {json.dumps(data5, indent=4)}")
    else:
        print(f"    Error: {r5.text[:400]}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
