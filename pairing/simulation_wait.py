"""Explicit passive physics waiting for the interactive learning runner.

Does not shorten appliance timers, select task actions, or advance physics while
an agent is thinking. Every tick still goes through the normal runtime/recorder.
"""
import copy
import time

import numpy as np

from roboswe.codex_login import DECISION_SCHEMA
from roboswe.library.attempts import AttemptRunner

MAX_WAIT_STEPS = 600
MAX_EPISODE_WAIT_STEPS = 1200
WAIT_SCHEMA = copy.deepcopy(DECISION_SCHEMA)
WAIT_SCHEMA['properties']['operation']['enum'].append('wait')
WAIT_INSTRUCTIONS = '''
Physics is paused between actions; real time spent reasoning does not cook food.
To deliberately let physics run while holding still, return operation="wait",
path="", content="500" (an integer control-step count), and your public rationale
in message. Each control step is 1/control_frequency_hz seconds. A wait may contain
1–600 steps, with at most 1,200 wait steps per episode; these count toward the 6,000
control-step limit. Wait requires a previous native step. It zeros arm/base/torso
motion commands and retains the last gripper targets. It does not finish a task
or detect appliance readiness for you. Check the returned sensors afterward.
Wait ticks are all recorded. Host time spent executing this passive wait is tracked
separately and excluded from the 720-second agent-work budget. Normal movement
executions retain the existing 50-step/60-second limits. Waiting is NOT permission
to send an extended movement sequence.
'''


def validate_wait_steps(steps, used=0):
    if type(steps) is not int or not 1 <= steps <= MAX_WAIT_STEPS:
        raise ValueError('Passive wait requires 1–600 integer control steps')
    if used + steps > MAX_EPISODE_WAIT_STEPS:
        raise ValueError('Episode passive-wait allowance exhausted')


def hold_action(session, previous):
    if previous is None:
        raise ValueError('Execute a native step before waiting so gripper targets are known')
    if hasattr(session, 'passive_hold_action'):
        return session.passive_hold_action(previous)
    schema = session.env.action_schema()
    controllers = schema['controllers']
    if len(controllers) != 1 or controllers[0]['controller'] != 'HybridMobileBase':
        raise ValueError('Passive hold has not been validated for this composite controller')
    controller = controllers[0]
    action = np.zeros(schema['shape'][0])
    for name, (start, end) in controller['parts'].items():
        if name.endswith('_gripper'):
            # Panda's native command is incremental: zero retains its formatted
            # target; repeating a close command would continue closing during wait.
            parameters = controller['parameters'].get(name, {})
            action[start:end] = 0 if parameters.get('gripper_class') == 'PandaGripper' else previous[start:end]
        elif name != 'base_mode':
            parameters = controller['parameters'][name]
            if parameters.get('input_type') != 'delta' and 'VelocityController' not in parameters.get('class', ''):
                raise ValueError(f'No validated passive hold for {name}')
    return action.tolist()


def passive_wait(runtime, previous, steps, *, on_step=lambda: None, clock=time.monotonic):
    used = getattr(runtime, 'passive_wait_steps', 0)
    validate_wait_steps(steps, used)
    if runtime.started is None or runtime.finished:
        raise ValueError('No active episode')
    if clock() - runtime.started >= runtime.timeout:
        raise ValueError('Agent-work budget exhausted before wait request')
    if runtime.session.steps + steps > runtime.max_steps:
        raise ValueError('Wait would exceed episode control-step budget')
    action = hold_action(runtime.session, previous)
    began = clock()
    advanced = 0
    for _ in range(steps):
        tick_start = clock()
        try:
            observation = runtime.bridge({'operation': 'step', 'action': action})
            advanced += 1
            runtime.passive_wait_steps = used + advanced
            on_step()
        finally:
            # Credit only trusted simulator work, not model time or normal motion.
            runtime.started += clock() - tick_start
        if observation.get('task_success') or observation.get('simulator_phase') != 'active':
            break
    elapsed = clock() - began
    runtime.passive_wait_wall_seconds = getattr(runtime, 'passive_wait_wall_seconds', 0.) + elapsed
    return {**observation, 'passive_wait': {'requested_steps': steps, 'advanced_steps': advanced,
            'host_seconds': elapsed, 'episode_wait_steps': runtime.passive_wait_steps}}


class WaitingAttemptRunner(AttemptRunner):
    def apply(self, decision, **kwargs):
        if not isinstance(decision, dict) or decision.get('operation') != 'wait':
            return super().apply(decision, **kwargs)
        if not self.task_active or set(decision) != {'operation', 'path', 'content', 'message'}:
            raise ValueError('Invalid wait decision')
        if not all(isinstance(v, str) for v in decision.values()) or decision['path'] or not decision['content'].isascii() or not decision['content'].isdigit():
            raise ValueError('Wait uses an empty path and a decimal integer in content')
        steps = int(decision['content'])
        validate_wait_steps(steps, self.runtime.passive_wait_steps)
        self.journal.append('operation_request', {'decision': decision, 'policy': 'passive_wait_v1'})
        result = self.runtime.wait(steps)
        self.journal.append('operation_result', {'operation': 'wait', 'result': {'control_step': result['control_step'], 'passive_wait': result['passive_wait']}, 'cumulative_action_sha256': self.runtime.action_digest.hexdigest()})
        return result
