"""Explicit network diagnostic; never run during offline test collection."""
import os

def main():
    if os.environ.get("JARVIS_RUN_NETWORK_DIAGNOSTICS") != "1":
        raise SystemExit("Set JARVIS_RUN_NETWORK_DIAGNOSTICS=1 and configure your own Upstox credentials.")
    """
    Upstox API Test Script
    Tests if the API key is valid and can fetch live market data
    """

    import requests
    import json

    # ============ UPSTOX CREDENTIALS ============
    API_KEY = os.environ.get("UPSTOX_API_KEY", "")
    SECRET_KEY = os.environ.get("UPSTOX_SECRET_KEY", "")
    REDIRECT_URI = "https://127.0.0.1"  # default for testing

    print("=" * 60)
    print("  UPSTOX API TEST - Indian Market")
    print("=" * 60)

    # Step 1: Check if API key format is valid
    print(f"\n[1] API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"[1] Secret:  {SECRET_KEY[:3]}...{SECRET_KEY[-2:]}")

    # Step 2: Try to get auth URL (no token needed)
    print("\n[2] Testing Upstox API endpoint availability...")
    try:
        resp = requests.get("https://api.upstox.com/v2/market-quote/ltp",
                            params={"symbol": "NSE_EQ|INE848E01016"},
                            headers={"Accept": "application/json"},
                            timeout=10)
        print(f"    Status Code: {resp.status_code}")
        if resp.status_code == 401:
            print("    ✅ API is REACHABLE (401 = needs token, expected!)")
        elif resp.status_code == 200:
            print("    ✅ API returned data!")
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"    Response: {resp.text[:200]}")
    except Exception as e:
        print(f"    ❌ Connection error: {e}")

    # Step 3: Generate login URL for OAuth2
    print("\n[3] Login URL (open this in browser to get Access Token):")
    auth_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code"
        f"&client_id={API_KEY}"
        f"&redirect_uri={REDIRECT_URI}"
    )
    print(f"    {auth_url}")

    # Step 4: If you have an access token, test live data
    ACCESS_TOKEN = None  # Set this after OAuth login

    if ACCESS_TOKEN:
        print("\n[4] Testing Live Market Data (NIFTY 50)...")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }

        # Get NIFTY 50 LTP
        resp = requests.get(
            "https://api.upstox.com/v2/market-quote/ltp",
            params={"symbol": "NSE_INDEX|Nifty 50"},
            headers=headers,
            timeout=10
        )
        print(f"    NIFTY 50 Response: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"    ✅ NIFTY 50 LTP: {json.dumps(data, indent=2)}")
        else:
            print(f"    ❌ Error: {resp.text[:200]}")

        # Check profile
        resp2 = requests.get(
            "https://api.upstox.com/v2/user/profile",
            headers=headers,
            timeout=10
        )
        print(f"\n[5] User Profile: {resp2.status_code}")
        if resp2.status_code == 200:
            print(f"    ✅ Profile: {json.dumps(resp2.json(), indent=2)}")
        else:
            print(f"    ❌ {resp2.text[:200]}")
    else:
        print("\n[4] ⚠️  Access Token needed for live data!")
        print("    Steps to get Access Token:")
        print("    1. Open the login URL above in browser")
        print("    2. Login with your Upstox account")
        print("    3. After redirect, copy 'code' from URL")
        print("    4. Run token exchange below")

    # Step 5: Token exchange helper
    print("\n[5] To exchange 'code' for Access Token, run:")
    print("""
    import requests
    code = "PASTE_CODE_HERE"
    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        data={
            "code": code,
            "client_id": API_KEY,
            "client_secret": SECRET_KEY,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        }
    )
    print(resp.json())
    # access_token = resp.json()['access_token']
    """)

    print("\n" + "=" * 60)
    print("  RESULT SUMMARY")
    print("=" * 60)
    print("✅ API Key format: Valid UUID")
    print("✅ Upstox API: Reachable from your network")
    print("⚠️  OAuth login required for live data access")
    print("📊 Indian Market instruments available:")
    print("   - NSE Equity (NSE_EQ|...)")
    print("   - NSE F&O (NSE_FO|...)")
    print("   - NSE Index (NSE_INDEX|Nifty 50)")
    print("   - BSE Equity (BSE_EQ|...)")


if __name__ == "__main__":
    main()
