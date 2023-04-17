"""This is the kondo_ml.py script."""


class KondoMLPackage:
    """Kondo-ML class."""

    def __init__(self, message=None):
        """Initialize with user-defined parameters."""
        self.message = message

    def show(self):
        """Run the show function."""
        if self.message is None:
            print("No message was given as input during initialization.")
        else:
            print(self.message)

        # Return
        return None
