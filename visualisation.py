__author__ = 'Tobias Kuhlmann'

'''
various plot functions to plot both wlsev and ols and realized for comparison
'''

def plot_results(X,Y,y_wlsev, y_ols):
    """
    plot results

    """
    import matplotlib

    matplotlib.use
    import matplotlib.pyplot as plt

    matplotlib.style.use('ggplot')

    plt.plot(range(0, len(y_ols)), y_ols,
             label='ols')
    plt.plot(range(0, len(y_wlsev)), y_wlsev,
             label='wls-ev')
    plt.plot(range(0, len(Y)), Y,
             label='realized')
    plt.title('OLS vs WLS-EV Time Series')
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.legend()
    plt.show()

def plot_results_custom(X, Y, y_wlsev, y_ols, title, ylabel):
    """
    plot results

    """
    import matplotlib

    matplotlib.use
    import matplotlib.pyplot as plt

    matplotlib.style.use('ggplot')

    plt.plot(range(0, len(y_ols)), y_ols,
             label='ols')
    plt.plot(range(0, len(y_wlsev)), y_wlsev,
             label='wls-ev')
    plt.plot(range(0, len(Y)), Y,
             label='realized')
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_scatter(X,Y,y_wlsev, y_ols):
    """
    plot scatter

    """
    import matplotlib

    matplotlib.use
    import matplotlib.pyplot as plt

    matplotlib.style.use('ggplot')

    plt.scatter(X, Y)
    plt.title('OLS vs WLS-EV Scatter')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plot wlsev prediction
    plt.plot(X, y_wlsev,
             label='wls-ev')
    # plot ols prediction
    plt.plot(X, y_ols,
             label='ols')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    """
    Get WLS-EV and OLS data
    """