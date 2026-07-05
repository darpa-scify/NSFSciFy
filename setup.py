import gdown
import os

download_id = "1GL3lDhOeh4ncFesCwrg1xTX7PJhN5-8_"
url = f"https://drive.google.com/uc?id={download_id}"
output = "dmr_nsf_awards_with_claims.jsonl.gz"
gdown.download(url, output, quiet=False)

os.makedirs("data", exist_ok=True)
os.system("gunzip dmr_nsf_awards_with_claims.jsonl.gz")
os.system("mv dmr_nsf_awards_with_claims.jsonl data/")
