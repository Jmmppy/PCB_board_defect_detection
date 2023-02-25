# -*- coding: utf-8 -*-
class glo:
    @staticmethod
    def _init():#初始化
        global _global_dict
        _global_dict = {}
    
    
    @staticmethod
    def set_value(key: str, value):
        """ 定义/改变一个全局变量 """
        # if glo.get_value(key, -1) == -1:
        _global_dict[key] = value
        # else:
        #     print(f"insert glo_dict {key}:{value} failed")
    
    
    @staticmethod
    def get_value(key: str, defValue=None):
        """ 获得一个全局变量,不存在则返回默认值 """
        return _global_dict.get(key, defValue)

    @staticmethod
    def print_glo():
        for k, v in _global_dict.items():
            print(f"({k} : {v}), ", end='')

if __name__ == '__main__':

    glo.print_glo()