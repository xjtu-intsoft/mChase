################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

from __future__ import print_function
import os, sys
import json
import sqlite3
import traceback
import argparse
import copy
import pandas as pd
import random

from urllib3 import Timeout
from async_timeout import timeout
import numpy as np
from tqdm.contrib import tzip
import eventlet

from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}


def condition_has_or(conds):
    return 'or' in conds[1::2]


def condition_has_like(conds):
    return WHERE_OPS.index('like') in [cond_unit[1] for cond_unit in conds[::2]]


def condition_has_sql(conds):
    for cond_unit in conds[::2]:
        val1, val2 = cond_unit[3], cond_unit[4]
        if val1 is not None and type(val1) is dict:
            return True
        if val2 is not None and type(val2) is dict:
            return True
    return False


def val_has_op(val_unit):
    return val_unit[0] != UNIT_OPS.index('none')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def accuracy(count, total):
    if count == total:
        return 1
    return 0


def recall(count, total):
    if count == total:
        return 1
    return 0


def F1(acc, rec):
    if (acc + rec) == 0:
        return 0
    return (2. * acc * rec) / (acc + rec)


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0,0,0
    elif count == pred_total:
        return 1,1,1
    return 0,0,0


def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1,1,1
    return len(pred_ao),len(label_ao),0


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def eval_nested(pred, label):
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += Evaluator().eval_exact_match(pred, label)
    return label_total, pred_total, cnt


def eval_IUEN(pred, label):
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], label['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], label['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], label['union'])
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt


def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


class Evaluator:
    """A simple evaluator"""
    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            if score['f1'] != 1:
                return 0
        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return label_tables == pred_tables
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1,'label_total':label_total,'pred_total':pred_total}

        return res


def isValidSQL(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except:
        return False
    return True


def print_scores(scores, etype):
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']

    print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    counts = [scores[level]['count'] for level in levels]
    print("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))

    if etype in ["all", "exec"]:
        print('=====================   EXECUTION ACCURACY     =====================')
        this_scores = [scores[level]['exec'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("execution", *this_scores))

    if etype in ["all", "match"]:
        print('\n====================== EXACT MATCHING ACCURACY =====================')
        exact_scores = [scores[level]['exact'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("exact match", *exact_scores))
        print('\n---------------------PARTIAL MATCHING ACCURACY----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        print('---------------------- PARTIAL MATCHING RECALL ----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        # print('---------------------- PARTIAL MATCHING F1 --------------------------')
        # for type_ in partial_types:
        #     this_scores = [scores[level]['partial'][type_]['f1'] for level in levels]
        #     print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))


def evaluate(gold, predict, db_dir, kmaps):
    # with open(gold) as f:
    #     glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    # with open(predict) as f:
    #     plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    
    # plist = [("select share, date, type from performance order by performance_id limit 2", "orchestra")]
    # glist = [("SELECT share, type FROM performance where performance_id = (select min(performance_id) from performance)", "orchestra")]
    plist = [('SELECT Count(*) FROM 汽车信息 JOIN 汽车名称 ON 汽车信息.id = 汽车名称.制造商ID JOIN 汽车制造商 ON 汽车名称.制造商ID = 汽车制造商.id WHERE 汽车制造商.制造商名称 = "American Motor Company"', "car_1")]
    glist = [('select count(*) from 汽车制造商 as t1 join 型号清单 as t2 on t1.ID = t2.制造商 where t1.制造商名称 = "American Motor Company"', "car_1")]
    evaluator = Evaluator()
    
    eval_err_num = 0
    exact_match = 0
    formally_unmatch_exec_match = 0
    formally_unmatch_exec_unmatch = 0
    db_error = 0
    cnt = 0
    for p, g in tzip(plist, glist):
        cnt += 1
        print(cnt)
        p_str = p[0]
        g_str, db = g
        db_name = db
        db = os.path.join(db_dir, db, db + ".sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, g_str)

        try:
            p_sql = get_sql(schema, p_str)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
            }
            eval_err_num += 1
            print("eval_err_num:{}".format(eval_err_num))
            continue
        
        g_sql_1 = copy.deepcopy(g_sql)
        exact_score = evaluator.eval_exact_match(p_sql, g_sql_1)
        if exact_score:
            exact_match += 1
            continue

        exec_score = eval_exec_match(db, p_str, g_str, p_sql, g_sql, schema)
        if not exec_score:
            formally_unmatch_exec_unmatch += 1
            continue

        result = parse_db(db, p_sql, g_sql)
        if not result:
            db_error += 1
            continue
        
        content, foreign_map_idx, columns_lower, tables = result
        flag = 1
        for i in range(50):
            exec_score = eval_exec_match_(db, p_str, g_str, p_sql, g_sql, content, foreign_map_idx, columns_lower, schema, tables)
            if exec_score == "db_error":
                db_error += 1
                flag = 0
                break
            if not exec_score:
                formally_unmatch_exec_unmatch += 1
                flag = 0
                break
        if flag:
            formally_unmatch_exec_match += 1


    print('-------------result--------------')
    print(f'grammar mistake: {eval_err_num}')
    print(f'exact match: {exact_match}')
    print(f'formally unmatch but equal: {formally_unmatch_exec_match}')
    print(f'formally unmatch and unequal: {formally_unmatch_exec_unmatch}')
    print(f'db_error: {db_error}')

def eval_exec_result(p_res, q_res, p_keys, q_keys):
    p_df = pd.DataFrame(p_res, columns=p_keys)
    q_df = pd.DataFrame(q_res, columns=q_keys)

    p_df.sort_index(axis=1, inplace=True)
    q_df.sort_index(axis=1, inplace=True)
    q_df.sort_values(by=list(q_df.columns), inplace=True)
    try:
        p_df.sort_values(by=list(q_df.columns), inplace=True)
    except:
        return False
    p_df = p_df[q_df.columns]
    p_df.index = range(len(p_df.index))
    q_df.index = range(len(q_df.index))
    return p_df.equals(q_df)


def parse_db(db, pred, gold):
    conn = sqlite3.connect(db)  
    cursor = conn.cursor()
    
    if isinstance(pred['from']['table_units'][0][1], dict):
        pred = pred['from']['table_units'][0][1]
    if isinstance(gold['from']['table_units'][0][1], dict):
        gold = gold['from']['table_units'][0][1]

    tables = [tb[1].strip('_') for tb in pred['from']['table_units']]
    tables.extend([tb[1].strip('_') for tb in gold['from']['table_units']])

    for item in pred['where']:
        if item != "and" and item != "or" and isinstance(item[3], dict):
            tables.extend([tb[1].strip('_') for tb in item[3]['from']['table_units']])
    for item in gold['where']:
        if item != "and" and item != "or" and isinstance(item[3], dict):
            tables.extend([tb[1].strip('_') for tb in item[3]['from']['table_units']])
    tables = list(set(tables))

    tables_join = ' join '.join(tables)

    try:
        cursor.execute(f'select * from {tables_join}')
        with eventlet.Timeout(3):
            content = cursor.fetchall()
    except Exception as :
        content = cursor.fetchmany(1000)
    except:
        return False

    def foreign_key_map(tables):
        map_ = {}
        for table in tables:
            cursor.execute(f"select * from pragma_foreign_key_list('{table}')")
            res = cursor.fetchall()
            for item in res:
                if item[2] in tables:
                    map_[f"{table}.{item[3]}"] = f"{item[2]}.{item[4]}"
        return map_

    foreign_map = foreign_key_map(tables)
    columns = {}
    columns_lower = {}
    id = 0
    for table in tables:
        cursor.execute(f"pragma table_info({table})")
        res = cursor.fetchall()
        for item in res:
            columns[f"{table}.{item[1]}"] = id
            columns_lower[f"{table.lower()}.{item[1].lower()}"] = id
            id += 1

    foreign_map_idx = {}
    for key in foreign_map:
        foreign_map_idx[columns[key]] = columns[foreign_map[key]]
    
    return content, foreign_map_idx, columns_lower, tables



def eval_exec_match_(db, p_str, g_str, pred, gold, content, foreign_map_idx, columns_lower, schema, tables):
    insert_list = []
    for i in range(100):
        insert_item = []
        max_len = len(content)
        for j in range(len(content[0])):
            random_element = content[random.randint(0, max_len - 1)][j]
            insert_item.append(random_element)
        insert_list.append(insert_item)

    for i in range(int(len(insert_list) * 0.8)):
        for key in foreign_map_idx:
            insert_list[i][key] = insert_list[i][foreign_map_idx[key]]
    random.shuffle(insert_list)

    where_conds = gold["where"]
    sample_num = [-10, -1, -0.1, 0, 0.1, 1, 10]
    for where_cond in where_conds:
        if where_cond != 'and' and where_cond != 'or' and where_cond[2][0] == 0:
            if isinstance(where_cond[3], dict):
                tmp = where_cond[3]
                while "where" in tmp:
                    tmp = tmp["where"]
                where_conds.extend(tmp)
                continue
    
            where_idx = columns_lower[where_cond[2][1][1].strip('_')]
            where_value = where_cond[3]

            for i in range(len(insert_list)):
                if insert_list[i][where_idx] == where_value and isinstance(where_value, str):
                    insert_list[i][where_idx] = 'undefined'
                if insert_list[i][where_idx] == where_value and not isinstance(where_value, str):
                    insert_list[i][where_idx] = -1
            for i in range(int(len(insert_list) * 0.4)):
                if isinstance(where_value, str):
                    insert_list[i][where_idx] = where_value.strip('"').strip("'")
                if not isinstance(where_value, str):
                    try:
                        insert_list[i][where_idx] += random.choice(sample_num)
                    except:
                        return "db_error"
        random.shuffle(insert_list)

    def insert_db(cursor, insert_list, tables):
        start_ = 0
        insert_list = np.array(insert_list, dtype=object)
        for table in tables:
            cursor.execute(f"pragma table_info({table})")
            res = cursor.fetchall()
            length = len(res)

            insert_item = insert_list[:, start_: start_ + length].tolist()
            start_ += length
            cursor.executemany(f'replace into {table} values ({"?,"*(length-1) + "?"})', insert_item)
        return cursor
    
    conn = sqlite3.connect(db)    
    cursor = conn.cursor()
    cursor = insert_db(cursor, insert_list, tables)
 
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except:
        return False

    cursor.execute(g_str)
    q_res = cursor.fetchall()
    
    p_cols = list(map(str, pred['select'][1]))
    q_cols = list(map(str, gold['select'][1]))
    p_cols_ = []
    q_cols_ = []

    ## 处理column为*的情况
    for p_col in p_cols:
        if "__all__" not in p_col or eval(p_col)[0] != 0:
            idx = 0
            while True:
                if p_col + str(idx) not in p_cols_:
                    p_cols_.append(p_col + str(idx))
                    break
                else:
                    idx += 1
        else:
            p_tables = [tb[1].strip('_') for tb in pred['from']['table_units']]
            p_cols_.extend(get_cols_from_tables(p_tables, schema, p_cols_))

    for q_col in q_cols:
        if "__all__" not in q_col or eval(q_col)[0] != 0:
            idx = 0
            while True:
                if q_col + str(idx) not in q_cols_:
                    q_cols_.append(q_col + str(idx))
                    break
                else:
                    idx += 1
        else:
            q_tables = [tb[1].strip('_') for tb in gold['from']['table_units']]
            q_cols_.extend(get_cols_from_tables(q_tables, schema, q_cols_))
    # print(p_res)
    # print(q_res)
    # print(p_cols_)
    # print(q_cols_)
    return eval_exec_result(p_res, q_res, p_cols_, q_cols_)

def get_cols_from_tables(tables: list, schema, p_cols_):
    cols = []
    for table in tables:
        assert table in schema.schema
        for col in schema.schema[table]:
            item = f"(0, (0, (0, '__{table}.{col}__', False), None))"
            idx = 0
            while True:
                if item + str(idx) not in p_cols_ and item + str(idx) not in cols:
                    cols.append(item + str(idx))
                    break
                else:
                    idx += 1
    return cols

def eval_exec_match(db, p_str, g_str, pred, gold, schema):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except:
        return False

    cursor.execute(g_str)
    q_res = cursor.fetchall()
    
    p_cols = list(map(str, pred['select'][1]))
    q_cols = list(map(str, gold['select'][1]))
    p_cols_ = []
    q_cols_ = []

    ## 处理column为*的情况
    for p_col in p_cols:
        if "__all__" not in p_col or eval(p_col)[0] != 0:
            idx = 0
            while True:
                if p_col + str(idx) not in p_cols_:
                    p_cols_.append(p_col + str(idx))
                    break
                else:
                    idx += 1
        else:
            p_tables = [tb[1].strip('_') for tb in pred['from']['table_units']]
            p_cols_.extend(get_cols_from_tables(p_tables, schema, p_cols_))

    for q_col in q_cols:
        if "__all__" not in q_col or eval(q_col)[0] != 0:
            idx = 0
            while True:
                if q_col + str(idx) not in q_cols_:
                    q_cols_.append(q_col + str(idx))
                    break
                else:
                    idx += 1
        else:
            q_tables = [tb[1].strip('_') for tb in gold['from']['table_units']]
            q_cols_.extend(get_cols_from_tables(q_tables, schema, q_cols_))
    return eval_exec_result(p_res, q_res, p_cols_, q_cols_)


# Rebuild SQL functions for value evaluation
def rebuild_cond_unit_val(cond_unit):
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_val(condition):
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res


def rebuild_sql_val(sql):
    if sql is None or not DISABLE_VALUE:
        return sql

    sql['from']['conds'] = rebuild_condition_val(sql['from']['conds'])
    sql['having'] = rebuild_condition_val(sql['having'])
    sql['where'] = rebuild_condition_val(sql['where'])
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])

    return sql


# Rebuild SQL functions for foreign key evaluation
def build_valid_col_units(table_units, schema):
    col_ids = [table_unit[1] for table_unit in table_units if table_unit[0] == TABLE_TYPE['table_unit']]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units= []
    for value in schema.idMap.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_col(valid_col_units, condition, kmap):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition


def rebuild_select_col(valid_col_units, sel, kmap):
    if sel is None:
        return sel
    distinct, _list = sel
    new_list = []
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    if DISABLE_DISTINCT:
        distinct = None
    return distinct, new_list


def rebuild_from_col(valid_col_units, from_, kmap):
    if from_ is None:
        return from_

    from_['table_units'] = [rebuild_table_unit_col(valid_col_units, table_unit, kmap) for table_unit in from_['table_units']]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap):
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap)
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'], kmap)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_condition_col(valid_col_units, sql['having'], kmap)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap)

    return sql


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str)
    parser.add_argument('--pred', dest='pred', type=str)
    parser.add_argument('--db', dest='db', type=str)
    parser.add_argument('--table', dest='table', type=str)
    args = parser.parse_args()

    gold = args.gold
    pred = args.pred
    db_dir = args.db
    table = args.table

    kmaps = build_foreign_key_map_from_json(table)


    evaluate(gold, pred, db_dir, kmaps)