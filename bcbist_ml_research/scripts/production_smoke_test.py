import requests
import time
import sys

def smoke_test():
    url = "http://0.0.0.0:8000"

    # 1. Health Check
    print("Checking /health...")
    try:
        resp = requests.get(f"{url}/health")
        if resp.status_code == 200:
            print("Health OK.")
        else:
            print(f"Health FAILED: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Connection FAILED: {str(e)}")
        return False

    # 2. Daily Picks Check
    print("Checking /api/research/predictions/daily-picks...")
    try:
        resp = requests.get(f"{url}/api/research/predictions/daily-picks")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Prediction SUCCESS. Selected K: {data.get('selected_k')}")
            print(f"Market Regime: {data.get('market_regime')}")
            if data.get('selected_k', 0) > 0:
                print(f"Top Pick: {data['predictions'][0]['symbol']} ({data['predictions'][0]['probability']:.2%})")
        else:
            print(f"Prediction API FAILED: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Prediction API EXCEPTION: {str(e)}")
        return False

    print("\nSMOKE TEST PASSED.")
    return True

if __name__ == "__main__":
    # Note: Assumes server is running (e.g. via start.sh)
    if smoke_test():
        sys.exit(0)
    else:
        sys.exit(1)
