from ADFM.task.train_dy import train_dy
from ADFM.task.train_edge import train_edge
from ADFM.task.gen_data import gen_data
from ADFM.task.plan import plan
from ADFM.task.train_critic import train_critic
from ADFM.task.plan_real import plan_real

def task_loader(task_name, real_robot=False):
    tasks = {
        'gen_data': gen_data,
        'train_dy': train_dy,
        'train_edge.yaml': train_edge,
        'train_critic': train_critic,
        'plan': plan if not real_robot else plan_real,
    }

    if task_name not in tasks:
        raise ValueError(f"Invalid task: {task_name}")
    else:
        return tasks[task_name]
