from conftest import make_traj
from loopstop.replay import crossfit_best_heuristic, task_fold


def _task_ids_in_both_folds():
    found = {}
    i = 0
    while len(found) < 2:
        task_id = f"selection-task-{i}"
        found.setdefault(task_fold(task_id), task_id)
        i += 1
    return found[0], found[1]


def test_best_heuristic_is_selected_out_of_fold():
    fold0_task, fold1_task = _task_ids_in_both_folds()
    trajs = [
        make_traj([1.0, 0.0], traj_id="fold0", task_id=fold0_task),
        make_traj([0.0, 1.0], traj_id="fold1", task_id=fold1_task),
    ]
    specs = [
        {"type": "fixed_n", "label": "n1", "params": {"n": 1}},
        {"type": "fixed_n", "label": "n2", "params": {"n": 2}},
    ]

    result = crossfit_best_heuristic(trajs, specs, lam=0.0, mode="stop_at_last")

    assert result.fold_winners == {0: "n2", 1: "n1"}
    assert result.mean_utility == 0.0
    assert result.per_traj_utility == {"fold0": 0.0, "fold1": 0.0}
