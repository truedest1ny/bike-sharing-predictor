from starter.starter import Starter


def main():
    """
        The main entry point of the application.

        This function triggers the Starter service, which orchestrates
        the data processing and machine learning workflows.
    """
    Starter.start()


if __name__ == '__main__':
    main()
