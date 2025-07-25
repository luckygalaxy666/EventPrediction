# -*- coding: utf-8 -*-
# from config.constants import search
from elasticsearch_dsl import Q
from elasticsearch import ConnectionError, ConnectionTimeout
from elasticsearch import NotFoundError, RequestError
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl import A
import traceback



def getkeywordsFunctionMessage(page_content, type_index, page_num, page_size):
    """
    根据fullname进行筛选
    :param message:
    :return:
    """
    start_from = ((page_num - 1) * page_size)
    if (start_from + page_size) > 10000:
        end_from = 10000
    else:
        end_from = start_from + page_size
    client = Elasticsearch(
    ['http://121.48.163.69:9200'],
    timeout=100,
    http_auth=('elastic', 'uestc123')
)
    search = Search(using=client, index=type_index)
    response_result = {}
    newData = []
    try:
        domain = {'page_content': page_content}
        qfullName = Q("match", **domain)

        print(qfullName)
        s = search
        s = s.query(qfullName)
        s = s[start_from:end_from]
        print(s)
        try:
            response = s.execute()
        except Exception as e:
            print("执行查询时发生错误:")
            print(traceback.format_exc())
    # 其他处理逻辑
        count2 = response.hits.total.value
        count1 = len(response.hits.hits)
        # print response
        for hit in response:
            newData.append(hit.to_dict())
        if count2 > 15:
            countNew = count2
        else:
            countNew = count1
    except ConnectionTimeout as e:
        print("连接超时，请确认网络状态和IP是否正确. \n\t\t错误: ",
              str(e.info.__class__.__name__), ",\n\t\t详细信息: ",
              str(e.info))
        info = {}

        info['msg'] = e.info
        response_result = {}
        response_result['message'] = str(e.info)
        response_result['code'] = 500
        response_result['data'] = []
        return None, response_result
    except ConnectionError as e:
        print("连接异常，请确认网络状态、IP和端口是否正确. \n\t\t错误: ",
              str(e.info.__class__.__name__), ",\n\t\t详细信息: ",
              str(e.info))
        info = {}

        info['msg'] = e.info
        response_result = {}
        response_result['message'] = str(e.info)
        response_result['code'] = 500
        response_result['data'] = []
        return None, response_result
    except NotFoundError as e:
        print("查询索引不存在，请确认所使用的索引是不是正确. \n\t\t状态码",
              str(e.status_code), ",\n\t\t错误: ",
              str(e.error), ",\n\t\t详细信息: ",
              str(e.info['error']['root_cause'][0]['reason']))
        info = {}

        info['msg'] = e.info
        response_result['message'] = str(e.info)
        response_result['code'] = 500
        response_result['data'] = []
        return None, response_result
    except RequestError as e:
        print("查询方法有问题，请确认queryMethod是否正确使用. \n\t\t状态码",
              str(e.status_code), ",\n\t\t错误: ",
              str(e.error), ",\n\t\t详细信息: ",
              str(e.info['error']['root_cause'][0]['reason']))
        info = {}

        info['msg'] = e.info
        response_result['message'] = str(e.info)
        response_result['code'] = 500
        response_result['data'] = []
        return None, response_result
    else:
        response_result['message'] = "查询成功"
        response_result['code'] = 200
        response_result['data'] = {"allData": newData, "allCount": countNew}
    finally:
        return response_result



if __name__ == "__main__":
    # 表T-bbs-keywords对照index的是bbsnewskeywords
    # 表T-news-keywords对照index的是newskeywords
    result = getkeywordsFunctionMessage("美国对台军售", "bbsnewskeywords", 99, 100)
    
    print(result['data']['allCount'])
    for i in range(0, 15):
        print (result['data']['allData'][i])
        print("-----------------")
        # print (result['data']['allData'][i]['date1'])
        # print (result['data']['allData'][i]['page_content'])
        # print (result['data']['allData'][i]['relations'])
    # print (result['data']['allData'][0]['date1'])
    # print (result['data']['allData'][0]['page_content'])
    # print (result['data']['allData'][0]['relations'])
