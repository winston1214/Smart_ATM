import matplotlib.pyplot as plt
def make_plot(ls,title):
    plt.figure(figsize=(12,8))
    plt.title(title)
    plt.plot(ls)
    plt.savefig(f'C:\\Users\\winst\\Desktop\\github_smartatm\\facial_recognition\\{title}.png')

ls = [1,2,3,4,5]
make_plot(ls,'sample')
