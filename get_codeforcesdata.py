from bs4 import BeautifulSoup as bs
import urllib.request, urllib.error
import csv
import time
import random
import unicodedata
# import pandas as pd
#
#
# # 创建一个 DataFrame 并指定列标题
# data = {'who': [1, 2, 3], 'problem': [4, 5, 6],'when': [1, 2, 3], 'try': [4, 5, 6],'problem_len': [1, 2, 3], 'input_len': [4, 5, 6],'output_len': [1, 2, 3], 'diff': [4, 5, 6],'result': [1, 2, 3]}
# df = pd.DataFrame(data)
#
# # 保存到 CSV 文件
# df.to_csv('output.csv', index=False)


def filter_non_utf8(string):
  filtered_string = ''
  for char in string:
    if unicodedata.category(char) != 'Other':
      filtered_string += char

  return filtered_string

def ask_url(url):
    head = {  # 模拟浏览器头部信息，向服务器发送消息
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    }
    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html


def get_data(data, url):
    t = 1
    for i in range(1,30):
        url = url.split('?')[0][0:48] + str(i) + '?' + url.split('?')[1]

        print(url)
        html = ask_url(url)
        soup = bs(html, "html.parser")

        for item in soup.find_all("tr"):
            c_data = item.get_text()
            # 去除无用符号
            c_data = c_data.replace('\n', '')
            c_data = c_data.replace('\r', '')
            c_data = c_data.replace(' ', '')
            # 记录提交数据
            if c_data[0].isdigit():
                c_datas = []
                # 提交时间
                c_datas.append(c_data[9:25])
                # 题号
                for j in range(0, len(c_data)):
                    if c_data[j] == '-' and c_data[j-1].isupper():
                        # 根据外国人的命名习惯 - 前的大写字母为题号
                        c_datas.append(c_data[j-1])
                        # 提交者
                        c_datas.append(filter_non_utf8(c_data[25:j-1]))
                        break
                    j += 1
                # 结果
                if "Accepted" in c_data:
                    c_datas.append("1")
                else:
                    c_datas.append("0")
                data.append(c_datas)

        filename = 'Codeforces 948.csv'
        # 每爬一页就将二维数组写入 CSV 文件，并采用 UTF-8 编码
        with open(filename, mode='a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in data:
                csv_writer.writerow(row)
        data.clear()
        # 十分粗糙且失败的反识别技术，随机休息一段时间，实际用下来还是会被发现。。
        t += 1
        if t < 50:
            time.sleep(random.uniform(0.225, 0.425))
        else:
            time.sleep(0.8)
            t = 1
# 数据
data = []
url = "https://codeforces.com/contest/1977/status/page/1?order=BY_ARRIVED_DESC"
# print(ask_url(url))
get_data(data, url)

