import yaml


class parse(object):
    """读取 YAML，并允许使用 params['Network'] 形式访问顶层字段。"""
    def __init__(self, path):
        with open(path, 'r') as file:
            self.parameters = yaml.safe_load(file)

    # Allow dictionary like access
    def __getitem__(self, key):
        return self.parameters[key]

    def save(self, filename):
        """把当前参数重新序列化到 YAML；训练主流程目前未调用此方法。"""
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)
