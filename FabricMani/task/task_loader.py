from FabricMani.task.train_dy import train_dy
from FabricMani.task.train_edge import train_edge
from FabricMani.task.gen_data import gen_data
from FabricMani.task.plan import plan
from FabricMani.task.plan_real import plan_real


def task_loader(task_name, real_robot=False):
    tasks = {
        'gen_data': gen_data,
        'train_dy': train_dy,
        'train_edge': train_edge,
        'plan': plan if not real_robot else plan_real,

    }

    if task_name not in tasks:
        raise ValueError(f"Invalid task: {task_name}")
    else:
        return tasks[task_name]
