import os
import re
from time import sleep
from typing import List, Dict


class ptr_color:
    notice: str = "\033[38;2;150;150;255m"
    flag: str = "\033[38;2;0;255;0m"
    info: str = "\033[38;2;255;255;255m"
    warning: str = "\033[38;2;255;165;0m"
    error: str = "\033[38;2;255;0;0m"

    pending: str = "\033[38;2;255;255;150m"
    training: str = "\033[38;2;150;255;150m"
    validating: str = "\033[38;2;0;255;255m"

    done: str = "\033[38;2;128;128;128m"


class scroll_template_list:
    def __init__(self, template_list: List[str]):
        self.pos: int = 0
        self.template_list = template_list
        self.value_list = ["" for i in range(len(template_list))]

    def roll(self, value: str):
        if self.pos < len(self.value_list):
            self.value_list[self.pos] = value
            self.pos += 1
        else:
            del self.value_list[0]
            self.value_list.append(value)
        return dict(zip(self.template_list, self.value_list))


class printer_template:
    def __init__(self) -> None:
        self.template = ""
        self.slot_set: List[str] = self.get_all_placeholder(self.template)
        self.length_dict: Dict[str, int] = {key: 0 for key in self.slot_set}
        self.value_dict: Dict[str, str] = {key: "" for key in self.slot_set}
        self.color_dict: Dict[str, str | None] = {key: None for key in self.slot_set}
        self.color_dict["_default"] = ptr_color.info
        self.color_dict["_reset"] = ptr_color.info
        self.value_t: Dict[str, str] = {}
        self.scroll_dict: Dict[str, scroll_template_list] = {}

    def add_template(self, template: str):
        keys = self.get_all_placeholder(template)
        self.slot_set.extend(keys)
        for key in keys:
            self.length_dict[key] = 0
            self.value_dict[key] = ""
            self.color_dict[key] = None
        self.template += template

    def set_value_template(self, key, template: str):
        self.value_t[key] = template

    def get_all_placeholder(self, template):
        pattern = re.compile(r"\{(.+?)\}")
        placeholders = pattern.findall(template)
        return placeholders

    def set_length(self, key: str, length: int):
        self.length_dict[key] = length

    def set_value_batch(
        self, v_dict: Dict[str, str | int | float | Dict[str, str | int | float]]
    ):
        for key, item in v_dict.items():
            self.set_value(key, item)

    def set_value(
        self, key: str, value: str | int | float | Dict[str, str | int | float]
    ):
        if key in self.value_t and not isinstance(value, Dict):
            value = self.value_t[key].format(**{"value": value})
        if isinstance(value, Dict):
            if key in self.value_t:
                value = self.value_t[key].format(**value)
            else:
                raise ValueError(f"key error:{key}")
        value = str(value)
        length = self.length_dict[key]
        v_l = len(value)
        if v_l > length:
            raise ValueError(f"key error:{key}")
        self.value_dict[key] = value

    def set_color(self, key: str, color: str | None):
        if color is not None:
            self.color_dict[key] = color

    def set_slot(self, key: str, value: str, length: int, color: str | None = None):
        self.set_length(key, length)
        self.set_color(key, color)
        self.set_value(key, value)

    def set_slot_t(
        self, key: str, template: str, length: int, color: str | None = None
    ):
        self.set_length(key, length)
        self.set_color(key, color)
        self.set_value(key, template)
        self.set_value_template(key, template)

    def w_flush(self, key: str, value: str | int | float, color: str | None = None):
        self.set_value(key, value)
        self.set_color(key, color)
        self.flush()

    def write(self, key: str, value: str | float | int, color: str | None = None):
        self.set_color(key, color)
        self.set_value(key, value)

    def add_scroll(self, key: str, template_list: List[str]):
        for s in template_list:
            if s not in self.slot_set:
                raise KeyError(f"key {s} not exist in slot_set")

        self.scroll_dict[key] = scroll_template_list(template_list)

    def scroll(self, key: str, value: str | float | int, color: str | None = None):
        ts = self.scroll_dict[key].roll(str(value))
        last_key: str = ""
        for k, v in ts.items():
            self.write(k, v)
            if v != "":
                last_key = k
        self.set_color(last_key, color)

    def scl_flush(self, key: str, value: str | float | int, color: str | None = None):
        self.scroll(key, value, color)
        self.flush()

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def flush(self):
        self.clear_screen()
        fomat_set = {}
        for key, value in self.value_dict.items():
            color = None
            prefix = ""
            suffix = ""
            value_format = "{prefix}{start}{value}{end}{suffix}"
            if self.color_dict[key] is None:
                color = self.color_dict["_default"]
            else:
                color = self.color_dict[key]
            length = self.length_dict[key]
            v_l = len(value)
            if v_l <= length:
                prefix_length = (length - v_l) // 2
                suffix_length = length - prefix_length - v_l
                prefix = " " * prefix_length
                suffix = " " * suffix_length

            fomat_set[key] = value_format.format(
                **{
                    "start": color,
                    "value": value,
                    "end": self.color_dict["_reset"],
                    "prefix": prefix,
                    "suffix": suffix,
                }
            )
        print(self.template.format(**fomat_set))


train_ptr = printer_template()
train_ptr.add_template(" " * 150 + "\n")
train_ptr.add_template(" " * 150 + "\n")
train_ptr.add_template("-" * 150 + "\n")
train_ptr.add_template("{name}{epochs}{encoder}{classifier}{lr}" + "\n")
train_ptr.add_template("-" * 150 + "\n")
train_ptr.add_template("{data_source}{split_type}{seed}{device}{resume}" + "\n")
train_ptr.add_template("-" * 150 + "\n")
train_ptr.add_template("{epoch}{current_lr}{elapsed}{early_stop}{state}" + "\n")
train_ptr.add_template("-" * 150 + "\n")


train_ptr.add_template("{i_1}{m_1}" + "\n")
train_ptr.add_template("{i_2}{m_2}" + "\n")
train_ptr.add_template("{i_3}{m_3}" + "\n")
train_ptr.add_template("{i_4}{m_4}" + "\n")
train_ptr.add_template("{i_5}{m_5}" + "\n")

train_ptr.add_template("-" * 150 + "\n")
train_ptr.add_template("{row_1}{train}" + "\n")
train_ptr.add_template("-" * 150 + "\n")
train_ptr.add_template("{row_2}{val}" + "\n")
train_ptr.add_template("-" * 150 + "\n")
train_ptr.add_template("{row_3}{best}" + "\n")
train_ptr.add_template("-" * 150 + "\n")


train_ptr.set_slot_t("name", "Name:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("epochs", "Total Epoch:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("encoder", "Encoder:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("classifier", "Classifier:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("lr", "LR:{value}", 30, ptr_color.info)


train_ptr.set_slot_t("data_source", "Data Source:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("split_type", "Split Type:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("seed", "Seed:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("device", "Device:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("resume", "Resume:{value}", 30, ptr_color.info)


train_ptr.set_slot_t("epoch", "Epoch:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("current_lr", "Current LR:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("elapsed", "Elapsed:{value} s", 30, ptr_color.info)
train_ptr.set_slot_t("early_stop", "Early Stop:{value}", 30, ptr_color.info)
train_ptr.set_slot_t("state", "State:{value}", 30, ptr_color.pending)


train_ptr.set_slot("row_1", "Train", 30, ptr_color.training)
train_ptr.set_slot("row_2", "Last Epoch Val", 30, ptr_color.validating)
train_ptr.set_slot("row_3", "Last Best Val", 30, ptr_color.validating)

train_ptr.set_slot("train", "Train", 120, ptr_color.training)
train_ptr.set_slot("val", "Last Epoch Val", 120, ptr_color.validating)
train_ptr.set_slot("best", "Last Best Val", 120, ptr_color.pending)

train_ptr.set_slot("i_1", "", 30)
train_ptr.set_slot("i_2", "", 30)
train_ptr.set_slot("i_3", "Info", 30)
train_ptr.set_slot("i_4", "", 30)
train_ptr.set_slot("i_5", "", 30)

train_ptr.set_slot("m_1", "", 120)
train_ptr.set_slot("m_2", "", 120)
train_ptr.set_slot("m_3", "", 120)
train_ptr.set_slot("m_4", "", 120)
train_ptr.set_slot("m_5", "", 120)

train_ptr.add_scroll("info", ["m_1", "m_2", "m_3", "m_4", "m_5"])
