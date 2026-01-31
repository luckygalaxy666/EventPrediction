from elasticsearch import Elasticsearch, helpers
import csv
from pathlib import Path
# part_key_entities = ["台积电", "英伟达", "AI芯片", "半导体", "美国", "魏哲家"]
# key_entities = ["赖清德","赖清德","拜登","蔡英文","美国","中国"]
event_name = "台积电获得英伟达AI芯片订单"
part_key_entities = ["台积电", "英伟达", "AI芯片", "半导体", "美国", "魏哲家"]
key_entities = part_key_entities.copy()
def connect_elasticsearch(host='http://121.48.163.69:45696'):
    """
    连接到 Elasticsearch 实例。

    参数:
        host (str): Elasticsearch 的连接地址。

    返回:
        Elasticsearch: Elasticsearch 客户端实例。
    """
    try:
        # 添加超时设置和重试配置
        es = Elasticsearch(
            [host],
            timeout=30,  # 连接超时30秒
            max_retries=3,  # 最大重试3次
            retry_on_timeout=True,  # 超时后重试
            verify_certs=False  # 如果使用自签名证书，设为False
        )
        # 使用更长的超时时间进行ping测试
        if not es.ping(request_timeout=10):
            raise ConnectionError("无法连接到 Elasticsearch 实例。请检查连接地址和 Elasticsearch 是否正在运行。")
        return es
    except Exception as e:
        raise ConnectionError(f"连接 Elasticsearch 失败: {str(e)}。请检查连接地址 {host} 和 Elasticsearch 是否正在运行。")

def build_query(content, should_keywords, date_start, date_end):
    """
    构建 Elasticsearch 查询体。

    参数:
        content (str): 必须匹配的内容。
        should_keywords (list): 应该匹配的关键词列表。
        date_start (str): 日期范围的开始日期（格式: 'YYYY-MM-DD'）。
        date_end (str): 日期范围的结束日期（格式: 'YYYY-MM-DD'）。

    返回:
        dict: Elasticsearch 查询体。
    """
    return {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": content}}
                        ],
                        "should": [
                            {"term": {"keywords": {"value": keyword}}} for keyword in should_keywords
                        ],
                        "filter": [
                            {
                                "range": {
                                    "date": {
                                        "gte": date_start,
                                        "lte": date_end
                                    }
                                }
                            }
                            # } if date_start and date_end else {}
                        ],
                        
                    }
                }
            }
        },
        "min_score": 10  # 最低得分阈值
    }

def fetch_all_documents(es, index_name, query_body, scroll='2m', size=1000):
    """
    使用 Scroll API 获取所有符合条件的文档。

    参数:
        es (Elasticsearch): Elasticsearch 客户端实例。
        index_name (str): 索引名称。
        query_body (dict): 查询体。
        scroll (str): Scroll 上下文保持时间。
        size (int): 每批次获取的文档数量。

    返回:
        list: 所有符合条件的文档列表。
    """
    # 初始搜索请求
    response = es.search(
        index=index_name,
        body=query_body,
        scroll=scroll,
        size=size,
        track_total_hits=True  # 确保可以获取总命中数
    )

    scroll_id = response.get('_scroll_id')
    hits = response.get('hits', {}).get('hits', [])
    docs = hits

    # 使用 Scroll API 获取剩余文档
    while True:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll)
        scroll_id = response.get('_scroll_id')
        hits = response.get('hits', {}).get('hits', [])
        if not hits:
            break
        docs.extend(hits)

    # 清理 scroll 上下文
    es.clear_scroll(scroll_id=scroll_id)

    print(f"总命中数: {len(docs)}")
    return docs

def filter_documents_by_score(docs, min_score=10):
    """
    过滤掉 _score 小于 min_score 的文档。

    参数:
        docs (list): 文档列表，每个文档应包含 '_score' 字段。
        min_score (float): 最低得分阈值。

    返回:
        list: 过滤后的文档列表。
    """
    filtered = []
    for doc in docs:
        score = doc.get('_score', 0)
        if score >= min_score:
            filtered.append(doc['_source'])
    print(f"筛选后符合条件的文档数量: {len(filtered)}")
    return filtered

def write_relations_to_csv(data, filename):
    """
    将数据写入 CSV 文件，每行包含 relation 和 date。

    参数:
        data (list): 包含 'relations' 和 'date' 字段的文档列表。
        filename (str): 输出的 CSV 文件名。
    """
    if not data:
        print("没有数据可以写入 CSV 文件。")
        return

    # 确保目录存在
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # 打开 CSV 文件进行写入
    with file_path.open(mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        
        # 遍历每个文档
        for doc in data:
            date = doc.get('date', '')
            relations = doc.get('relations', [])
            
            if not relations:
                continue  # 如果没有 relations，则跳过该文档
            
            for relation in relations:
                first_entity = relation.get('firstEntity', '')
                rel_type = relation.get('relation', '')
                second_entity = relation.get('secondEntity', '')
                if len(first_entity)>10 or len(second_entity)>10:
                    continue
                flag1 = False
                flag2 = False
                for i in range(len(part_key_entities)):
                    if flag1 == False and part_key_entities[i] in first_entity:
                        first_entity = key_entities[i]
                        flag1 = True
                    if flag2 == False and part_key_entities[i] in second_entity:
                        second_entity = key_entities[i]
                        flag2 = True
                    if flag1 and flag2:
                        break
                # 写入一行数据
                writer.writerow([
                    first_entity, 
                    "UNK",
                    rel_type,
                    second_entity,   
                    "UNK",      
                    date
                ])
    
    print(f"数据已成功写入 '{filename}' 文件。")

def main():
    # 定义查询参数
    index = "tsmcnews"  # 替换为你的索引名称
    keywordS_dict = {
        # "台积电将生产英伟达AI芯片": ["台积电", "英伟达", "AI芯片", "半导体","美国","亚利桑那州","特朗普","魏哲家"]
        event_name: part_key_entities
        # "台积电拿下特斯拉辅助驾驶芯片大单": ["台积电", "特斯拉", "芯片", "半导体", "美国", "魏哲家","三星"],
    # "美国对台军售": ["拜登", "台湾", "美国", "美国政府", "郭雅慧", "军售"],
    # "赖清德纪念金门战役": ["赖清德", "金门", "中华民国", "总统", "台澎", "军事", "国家主权", "台海"],
    # "台方支持美军舰入境": ["美国", "海军", "加拿大", "海峡", "台海", "政府", "台湾", "民主", "和平"],
    # "联合利剑演习": ["赖清德", "总统", "演习", "空军", "海军", "国家主权", "军演", "安全", "军队", "台湾民众", "中国"],
    # "台洪断交": ["台湾", "洪都拉斯", "贸易", "中华民国", "赖清德", "总统", "卓荣泰", "中国"],
    # "台舰新建计划": ["吴钊燮", "赖清德", "总统", "军演", "中国", "演习", "海军", "卓荣泰", "海巡", "国安", "军事"],
    # "赖清德视察演习军队": ["赖清德", "总统", "军演", "空军", "海军", "人民", "台湾民众", "国家主权", "海域", "军队", "海巡", "吴钊燮", "顾立雄", "刘任远", "张忠龙"],
    # "赖清德与无国界记者组织会晤": ["赖清德", "总统", "记者", "亚太", "亚洲", "台湾", "民主", "自由", "媒体", "人民", "台湾民众"],
    # "赖清德接见瓜地马拉外宾": ["瓜地马拉", "总统", "记者", "亚太", "亚洲", "台湾", "民主", "自由", "媒体", "人民", "台湾民众"],
    # "蔡英文国庆发言": ["蔡英文", "前总统", "民主", "赖清德", "台湾", "世界", "人民", "自由", "国庆"],
    # "双十讲话": ["战役", "赖清德", "萧美琴", "卓荣泰", "吐瓦鲁", "台湾", "民主", "双十", "国家主权", "人民", "台湾民众", "总统府", "净零", "国防", "中国"],
    # "赖清德接见露西亚参议长": ["总统", "赖清德", "吐瓦鲁", "戴斐立", "吐瓦鲁", "萧美琴", "世界卫生大会", "联合国", "台湾", "双十", "林佳龙", "林东亨", "潘孟安"],
    # "赖清德支持访欧": ["总统", "赖清德", "欧洲", "经济", "雷诺", "议长", "露西亚", "法兰西斯", "台湾", "人民", "卫生", "合作", "林佳龙", "外交"],
    # "赖清德维护主权": ["赖清德", "国庆", "中华民国", "国家主权", "人民", "台湾民众", "卓荣泰", "李鸿钧", "徐佳青", "刘世芳", "韩国瑜", "林佳龙", "外交"],
    # "美国军事援助": ["拜登", "台湾", "美国", "军援", "白宫", "国防", "援助", "战略", "印太"],
    # "全社会防卫韧性": ["赖清德", "总统", "全社会防卫韧性委员会", "国际社会", "国家气候变迁对策委员会", "健康台湾推动委员会", "蔡英文", "前总统", "刘得金", "锺树明", "台湾资安", "第一岛链", "民主", "萧美琴", "潘孟安"],
    # "台方感谢军售": ["拜登", "军售", "美国", "高雄", "国军", "自由", "民主", "宪政体制", "国会", "郭雅慧", "台湾", "六项保证", "台美", "威权主义"],
    # "赖清德出席典礼": ["赖清德", "总统", "高雄", "国军", "自由", "民主", "典礼", "宪政体制", "国防", "顾立雄", "刘得金", "锺树明"],
    # "赖清德接见吐瓦鲁国会议长": ["赖清德", "总统", "吐瓦鲁", "伊塔雷理", "安全", "伙伴关係", "蔡英文", "前总统", "中华民国", "美国", "澳洲", "全球合作暨训练架构", "日本", "联合国", "世界卫生大会", "太平洋岛国论坛", "马前总统", "卫生部长", "林佳龙"],
    # "赖清德视察澎湖军队": ["赖清德", "总统", "官兵", "国家安全", "国军", "国防", "军队", "海军", "顾立雄", "刘得金", "空军", "部队", "台湾"],
    # "台政府参与联合国案": ["联合国", "一中", "中华人民共和国", "中华民国", "台湾", "台海", "印太", "永续发展目标", "古特雷斯", "林佳龙", "赖清德", "和平", "民主伙伴", "朝野立委", "美国", "对中", "中国"]

}
    for content, should_keywords in keywordS_dict.items():
        start_date = "2024-06-01"
        end_date = "2025-02-01"
        min_score = 10
        size = 1000  # 每批次获取的文档数量，根据需要调整
        es_host = "http://121.48.163.69:45696"
        csv_filename = "./tsmc_es_data/" + content.replace(" ", "_") + ".csv"

        # 连接到 Elasticsearch
        es = connect_elasticsearch(host=es_host)
        
        # 构建查询体
        query_body = build_query(content, should_keywords, start_date, end_date)

        # 执行查询并获取所有文档
        try:
            all_documents = fetch_all_documents(es, index, query_body, scroll='2m', size=size)
        except Exception as e:
            print(f"查询过程中出错: {e}")
            return

        # 过滤得分
        filtered_documents = filter_documents_by_score(all_documents, min_score=min_score)

        # 写入 CSV 文件
        write_relations_to_csv(filtered_documents, csv_filename)

if __name__ == "__main__":
    main()