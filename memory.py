import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, input_shapes):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=input_shapes['o'])
        self.actions = RingBuffer(limit, shape=input_shapes['u'])
        self.rewards = RingBuffer(limit, shape=(1,))
        self.goals = RingBuffer(limit, shape=input_shapes['g'])
        self.observations1 = RingBuffer(limit, shape=input_shapes['o'])

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries, size=batch_size)
        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        goals_batch = self.goals.get_batch(batch_idxs)

        result = {
            'o': array_min2d(obs0_batch),
            'o_2': array_min2d(obs1_batch),
            'r': array_min2d(reward_batch),
            'u': array_min2d(action_batch),
            'g': array_min2d(goals_batch),
        }
        return result

    def append(self, obs0, obs1, action, reward, goal, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.observations1.append(obs1)
        self.actions.append(action)
        self.rewards.append(reward)
        self.goals.append(goal)

    @property
    def nb_entries(self):
        return len(self.observations0)
