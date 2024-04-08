import json
# 省份、字母和广告码
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

provinces_after = []
alphabets_after = []
ads_after = []

# 加载 JSON 文件
with open('D:\My_Code_Project\三下机器学习课设\解压数据包\单数字\VehicleLicense\dataset_info.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
# 获取标签字典
label_dict = data['label_dict']

for province in provinces:
    for key, value in label_dict.items():
        if value == province:
            provinces_after.append(key)

for alphabet in alphabets:
    for key, value in label_dict.items():
        if value == alphabet:
            alphabets_after.append(key)

for ad in ads:
    for key, value in label_dict.items():
        if value == ad:
            ads_after.append(key)

print(provinces_after)
print(alphabets_after)
print(ads_after)
['53', '25', '28', '59', '27', '48', '36', '33', '30', '24', '47', '64', '29', '37', '19', '34', '58', '16', '55', '60', '22', '43', '13', '21', '61', '63', '46', '20', '42', '39', '56']
['10', '11', '12', '14', '15', '17', '18', '23', '26', '31', '32', '35', '38', '40', '41', '44', '45', '49', '50', '51', '52', '54', '57', '62']
['10', '11', '12', '14', '15', '17', '18', '23', '26', '31', '32', '35', '38', '40', '41', '44', '45', '49', '50', '51', '52', '54', '57', '62', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
