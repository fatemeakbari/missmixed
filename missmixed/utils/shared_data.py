class SharedData:
    _instance = None
    data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_value(self, key, value):
        self.data[key] = value

    def get_value(self, key):
        return self.data.get(key, "Not Found")

    def is_categorical(self):
        return self.data['categorical_columns'][self.data['processing_col_idx']]

    def num_of_class(self):
        return self.data['category_counts'][self.data['processing_col_idx']]