import json
import requests
import xmltodict
import pandas as pd

serviceKey = "sYm9at2ytlmuaxU0XnPLBcbTnHesnQFFjqS1WrznTEf5npt1h1YhQ1L57epEjhhQxkk4rQZJo0w8xQ6MVhMapQ=="

# url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcRHTrade?_wadl&type=xml"

# base_date = "202001"
# gu_code = "22060"

# payload = (
#     "LAWD_CD="
#     + gu_code
#     + "&"
#     + "DEAL_YMD="
#     + base_date
#     + "&"
#     + "serviceKey="
#     + serviceKey
#     + "&"
# )

# res = requests.get(url + payload)
# print(res)


def get_df(lawd_cd, deal_ymd):
    global serviceKey
    base_url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcRHTrade?"
    base_url += f"&LAWD_CD={lawd_cd}"
    base_url += f"&DEAL_YMD={deal_ymd}"
    base_url += f"&serviceKey={serviceKey}"

    res = requests.get(base_url)
    data = json.loads(json.dumps(xmltodict.parse(res.text)))
    check = data['response']
    print(f'{check}')
    # df = pd.DataFrame(data["response"]["body"]["items"]["item"])

    return check


data = get_df(27260, 202209)
