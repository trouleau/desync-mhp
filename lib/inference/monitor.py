from tick.solver import History


class Monitor(object):

    def __init__(self, verbose=False, print_every=-1, record_every=-1):
        self._verbose = verbose
        self._print_every = print_every
        self._record_every = record_every
        self._history = History()
        self._history._clear()

    def set_print_order(self, print_order):
        self._history.print_order = print_order

    def set_print_style(self, print_style):
        self._history._print_style.update(print_style)

    def print_init(self, name):
        if self._verbose:
            print(':: Launching {:s}...'.format(name))

    def receive(self, force_print=False, **kwargs):
        n_iter = kwargs['n_iter']
        verbose = self._verbose
        print_every = self._print_every
        record_every = self._record_every
        should_print = verbose and (force_print or n_iter % print_every == 0)
        should_record = force_print or \
            n_iter % print_every == 0 or \
            n_iter % record_every == 0
        if should_record:
            self._history._update(**kwargs)
        if should_print and self._verbose:
            self._history._print_history()
            print(flush=True, end='')

    def as_dict(self):
        return dict(self._history.values)

    @property
    def values(self):
        return self._history.values
