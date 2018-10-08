import gc
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import BSpline, splrep
from sklearn.tree import DecisionTreeClassifier, _tree


class FeaturesPlotAnalysis():

    def __init__(self, df, label="label"):
        self.df = df
        self.label = label

    def category_plot(self, vari_col, xlabel_angle=0):
        sns.set_style("whitegrid")
        ncount = self.df.shape[0]
        ax = sns.countplot(x=vari_col, hue=self.label, data=self.df[[vari_col, self.label]])
        ax2 = ax.twinx()
        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')
        ax2.set_ylabel('Frequency [%]')
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate(
                '{:.3f}%'.format(100. * y / ncount), (x.mean(), y),
                ha='center',
                va='bottom')
        ax2.set_ylim(0, 100)
        ax2.grid(None)
        plt.xticks(rotation=xlabel_angle)
        plt.title("Distribution of %s" % vari_col)
        plt.show()

    def category_cross_var_plot(self,
                                count_vari,
                                cat_vari,
                                xlabel_angle=0,
                                box_cox_cut=True):
        sns.set_style("whitegrid")
        df0 = self.df[[count_vari, cat_vari, self.label]]
        #     plt.subplot(121)
        #     ncount = df.shape[0]
        #     ax = sns.countplot(
        #         x=cat_vari,
        #         hue=flag_col,
        #         data=df0)

        #     ax2 = ax.twinx()
        #     ax2.yaxis.tick_left()
        #     ax.yaxis.tick_right()
        #     ax.yaxis.set_label_position('right')
        #     ax2.yaxis.set_label_position('left')
        #     ax2.set_ylabel('Frequency [%]')
        #     for p in ax.patches:
        #         x = p.get_bbox().get_points()[:, 0]
        #         y = p.get_bbox().get_points()[1, 1]
        #         ax.annotate(
        #             '{:.3f}%'.format(100. * y / ncount), (x.mean(), y),
        #             ha='center',
        #             va='bottom')
        #     ax2.set_ylim(0, 100)
        #     ax2.grid(None)
        #     plt.xticks(rotation=xlabel_angle)
        #     plt.title("Distribution of %s" % vari_col)
        #     plt.subplot(122)
        if box_cox_cut:
            q1 = df0[count_vari].quantile(0.25)
            q3 = df0[count_vari].quantile(0.75)
            iqr = q3 - q1
            up_outline = q3 + 1.5 * iqr
            low_outline = q1 - 1.5 * iqr

            def outline(x):
                if x > up_outline:
                    return up_outline + .5 * iqr
                elif x < low_outline:
                    return low_outline - .5 * iqr
                else:
                    return x

            df0[count_vari] = df0[[count_vari]].applymap(lambda x: outline(x))
        sns.boxplot(
            x=cat_vari, y=count_vari, hue=self.label, data=df0, palette="Set3")
        plt.xticks(rotation=xlabel_angle)
        gc.enable()
        del df0
        gc.collect()
        plt.show()

    def distribute_plot(self,
                        vari_col,
                        flag_value=[1, 0],
                        bin_num=10,
                        plt_chioce="kdeplot"):
        sns.set_style("whitegrid")
        fig, ax_hist = plt.subplots()
        fig.set_size_inches(10, 5)
        plt.title("Distribution of %s" % vari_col)

        if plt_chioce == "distplot":
            sns.distplot(
                self.df.loc[self.df[self.label] == flag_value[1], vari_col].dropna(),
                kde=True,
                bins=bin_num,
                color="lawngreen",
                label=1)
            sns.distplot(
                self.df.loc[self.df[self.label] == flag_value[0], vari_col].dropna(),
                kde=True,
                bins=bin_num,
                color="dodgerblue",
                label=0)
        else:
            sns.kdeplot(
                self.df.loc[self.df[self.label] == flag_value[0], vari_col].dropna(),
                shade=True,
                label=0)
            sns.kdeplot(
                self.df.loc[self.df[self.label] == flag_value[1], vari_col].dropna(),
                shade=True,
                label=1)
        plt.show()

    def hist_plot(self,
                  vari_col,

                  flag_value=[1, 0],
                  label=['1', '0'],
                  bin_num=5):
        sns.set_style("whitegrid")
        fig, ax_hist = plt.subplots()
        ax_hist2 = ax_hist.twinx()

        # chang the bin number
        bin = bin_num
        while Counter(np.histogram(self.df[vari_col].values, bin)[0])[0] > 0:
            z = Counter(np.histogram(self.df[vari_col].values, bin)[0])[0]
            bin = bin - z
        if bin > 0:
            n, bins, patches = ax_hist.hist(
                [
                    self.df.loc[self.df[self.label] == flag_value[0], vari_col],
                    self.df.loc[self.df[self.label] == flag_value[1], vari_col]
                ],
                rwidth=0.75,
                label=label,
                color=["steelblue", "burlywood"],
                alpha=0.75,
                bins=bin)
            ax_hist.set_ylabel("Numbers Count")

            y = []
            x = []
            for i in range(bins.size - 1):
                if (n[0][i] + n[1][i]) > 0:
                    y.append(n[0][i] / (n[0][i] + n[1][i]))
                    x.append((bins[i] + bins[i + 1]) / 2)
            if len(x) > 2:
                if bin > 5:
                    k = 5
                elif bin > 1:
                    k = bin - 1
                else:
                    k = 1
                t, c, k = splrep(x, y, s=0, k=k)
                xnew = np.linspace(min(x), max(x), bin * 5)
                power_smooth = BSpline(t, c, k)
                ax_hist2.grid(None)
                ax_hist2.set_yticks([])
                ax_hist2.plot(
                    x, y, 'rs', xnew, power_smooth(xnew), 'c', color="0.6")

            else:
                ax_hist2.grid(None)
                ax_hist2.set_yticks([])
                ax_hist2.plot(x, y, 'rs', x, y, 'c', color="0.6")

            # display the line value

            for i, b in enumerate(zip(x, y)):
                ax_hist2.text(b[0], b[1], str(round(b[1], 3)), fontsize=12)

            ncount = self.df[vari_col].shape[0]
            ax2 = ax_hist.twinx()
            ax2.yaxis.tick_left()
            ax_hist.yaxis.tick_right()
            ax_hist.yaxis.set_label_position('right')
            ax2.yaxis.set_label_position('left')
            ax2.set_ylabel('Frequency [%]')
            for p in ax_hist.patches:
                x = p.get_bbox().get_points()[:, 0]
                y = p.get_bbox().get_points()[1, 1]
                ax_hist.annotate(
                    '{:.3f}%'.format(100. * y / ncount), (x.mean(), y),
                    ha='center',
                    va='bottom')
            ax2.set_ylim(0, 100)
            ax2.grid(None)

            plt.xticks(bins)  # x axis
            ax_hist.legend()
            fig.set_size_inches(10, 5)
            plt.title("Distribution of %s" % vari_col)
            plt.show()

        else:
            print("Can not plot,as the bin num is zero.")

    def tree_bin_plot(self, vari_col, bins=5, xlabel_angle=0):
        x = self.df[vari_col]
        y = self.df[self.label]
        clf = DecisionTreeClassifier(
            criterion="entropy", max_depth=bins, max_leaf_nodes=bins)
        x = x.values.reshape(x.shape[0], 1)
        print("Training decision tree ...")
        clf.fit(x, y)
        count_leaf = 0
        for i in clf.tree_.children_left:
            if i == _tree.TREE_LEAF:
                count_leaf += 1

        threshold = clf.tree_.threshold  # 所有的节点全部是<=
        # threshold=np.sort(threshold)[count_leaf:]
        # #这种方式不太好，如果有小于-2的排序就会出问题。
        #  -2的数目和叶子数相同 后面需要排除掉-2的值
        count = 0
        for i in threshold:
            if i == -2: count += 1

        new_threshold = list(filter(lambda x: x != -2, threshold))

        if count > count_leaf: new_threshold += [-2]

        new_threshold_2 = np.sort(new_threshold)

        thres_index = np.asarray(new_threshold_2).searchsorted(
            x, side='right')  # 注意这里的 right 表示的是<=
        thres_index = thres_index.ravel()

        print("Transform data ...")

        dfin = self.df[[vari_col, self.label]]
        xmax = x.max()

        dfin["TreeBin"] = dfin.index.map(
            lambda x: new_threshold_2[thres_index[x]] if thres_index[x] + 1 <= len(new_threshold_2) else xmax
        )

        sns.set_style("whitegrid")

        ncount = dfin.shape[0]
        ax = sns.countplot(x="TreeBin", hue=self.label, data=dfin)

        ax2 = ax.twinx()
        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')
        ax2.set_ylabel('Frequency [%]')
        #     y_plt=[]
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            #         y_plt.append(y)
            ax.annotate(
                '{:.3f}%'.format(100. * y / ncount), (x.mean(), y),
                ha='center',
                va='bottom')

        #     y_plt2=[]
        #     for i in range(bins) :
        #         y_plt2.append(round(y_plt[i+bins]/y_plt[i],3)*100)

        #     print(np.append(new_threshold_2,[xmax]))

        #     ax3=ax2.twinx()
        #     ax3.plot(
        #         np.append(new_threshold_2,[xmax]),
        #         y_plt2)
        #     ax3.set_yticks([])
        #     ax3.grid(None)
        ax2.grid(None)

        plt.xticks(rotation=xlabel_angle)
        plt.title("Distribution of %s" % vari_col)
        gc.enable()
        del dfin
        gc.collect()
        plt.show()
