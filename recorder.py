
class Recorder:

    def __init__(self, params):

        self.path = params['agent']['performance_path']

    def record_json(self, metrics):
        """Save performance numbers."""
        try:
            json.dump(metrics, open(self.path, "w"), indent=2)
            print(f'saved model in {self.path}')
        except Exception as e:
            print(f'Error: Writing model fails: {self.path}')
            print(e)
