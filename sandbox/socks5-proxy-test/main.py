# pip install -U requests[socks]

import requests

resp = requests.get('http://jsonip.com', proxies=dict(http='socks5://127.0.0.1:9150')) #,
                                #  https='socks5://user:pass@host:port'))
print(resp.content)