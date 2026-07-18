from torch.utils.data import Dataset


class PromptDataset(Dataset):
    def __init__(self, args, raw_data, tokenizer):
        super().__init__()
        self.args = args
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.item_size = {}
        self.max_length = args.max_length

    def _get_rainbow_pad_ids(self):
        """Cache rainbow pad token IDs."""
        if not hasattr(self, '_rainbow_pad_ids'):
            self._rainbow_pad_ids = [
                self.tokenizer.convert_tokens_to_ids(f"<rpad_{i}>")
                for i in range(7)
            ]
        return self._rainbow_pad_ids

    def __len__(self):
        return len(self.raw_data)

    def size(self, index):
        if index not in self.item_size:
            item = self.__getitem__(index)
            self.item_size[index] = len(item["source"])
        return self.item_size[index]

    def ordered_indices(self):
        raise NotImplementedError

    @staticmethod
    def load_data(args, tokenizer, train=True, valid=True, test=False):
        raise NotImplementedError

    def build_prompt(self, item):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
