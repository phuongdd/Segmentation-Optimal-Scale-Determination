        # cv_mean = df.groupby(by=fields[0]).mean()
        # cv_mean = pd.DataFrame(np.c_[cv_mean, cv_mean.pct_change()], columns=['CV', 'RoC'], index=cv_mean.index)
        # cv_mean.sort_index(axis=0, inplace=True)

        # fig, ax = plt.subplots(figsize=(15, 8))
        # ax2 = ax.twinx()
        # l1 = ax.plot(cv_mean['CV'], '-r*', label='CV')
        # l2 = ax2.plot(cv_mean['RoC'], '--bo', label='RoC')
        # lns = l1+l2
        # labs = [l.get_label() for l in lns]
        # ax.legend(lns, labs, loc=8)
        # # ax.set_xticks(cv_mean.index)
        # # ax.set_xticklabels(cv_mean.index, rotation=0, fontsize=14)
        # plt.title(title)
        # plt.savefig(f"{self.results_dir}/cv_roc.{title}.png")
        # if show:
        #     plt.show()
        # plt.close()

        # ---------------------------------------

        # Combined Big Box Plot
        # fig, ax = plt.subplots(figsize=(20, 5))
        # sns.boxplot(data=df, x=fields[0], y='value', hue='group')
        # ax.set_xlabel('')
        # plt.title(title)
        # plt.tight_layout()
        # plt.savefig(f"{self.results_dir}/bigboxplot.{title}.png")
        # if show:
        #     plt.show()
        # plt.close()
        
        # pbar.update(1)

        # ---------------------------------------
        # Individual Histogram and Boxplot
        # for param in df[fields[0]].unique():
        #     df_param = df[df[fields[0]] == param]
        #     df_cv = df_param[df_param['group'] == 'cv'].drop(labels=['group', fields[0]], axis=1)
        #     df_ncv = df_param[df_param['group'] == 'ncv'].drop(labels=['group', fields[0]], axis=1)

        #     for name, _df in [('CV', df_cv), ('NCV', df_ncv)]:
        #         if len(fields) == 1:
        #             _title = f"{name} {title}, k={param}"
        #         elif len(fields) == 2:
        #             _title = f"{name} {title}, markers={param}"
        #             pass

        #         pbar.set_postfix(desc=f"{_title}")

        #         # Individual BoxPlot
        #         fig, ax = plt.subplots(figsize=(10, 5))
        #         sns.boxplot(data=_df, y='value', color='lightblue', orient='h')
        #         ax.set_xlabel('')
        #         plt.title(_title)
        #         plt.tight_layout()
        #         plt.savefig(f"{self.results_dir}/boxplot.{_title}.png")
        #         if show:
        #             plt.show()
        #         plt.close()
        #         pbar.update(1)

        #         # ---------------------------------------

        #         # Individual Histogram
        #         fig, ax = plt.subplots(figsize=(10, 5))
        #         sns.distplot(_df, kde=True, rug=False, bins=100, hist=True)

        #         _mean = df['value'].mean()
        #         _max = df['value'].max()
        #         _min = df['value'].min()
        #         _median = df['value'].median()

        #         ax.axvline(x=_mean, linewidth=1, color='green', ls='-.')
        #         ax.axvline(x=_median, linewidth=1, color='yellow', ls='--')
        #         ax.axvline(x=_max, linewidth=1, color='red', ls='-')
        #         ax.axvline(x=_min, linewidth=1, color='blue', ls='-')

        #         s = f"Min: {_min:.4f}\n"
        #         s += f"Max: {_max:.4f}\n"
        #         s += f"Avg: {_mean:.4f} \n"
        #         s += f"Median: {_median:.4f}"
        #         ax.text(x=0.6,
        #                 y=0.9,
        #                 s=s,
        #                 va='top',
        #                 bbox=dict(boxstyle='round4', edgecolor=(0,0,0,1), fc=(1,1,1,1)),
        #                 transform=ax.transAxes
        #                 )

        #         ax.set_title(_title)
        #         ax.set_xlabel('')
        #         ax.grid('on')

        #         plt.tight_layout()
        #         plt.savefig(f"{self.results_dir}/histogram.{_title}.png")
        #         if show:
        #             plt.show()
        #         plt.close()
        #         pbar.update(1)
