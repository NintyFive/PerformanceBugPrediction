library(beanplot)
library(effsize)

issues = read.csv(file = 'issue_characteristics.csv')
tmp_issues = issues[issues$resolve_time > 0, ]
general_bugs = tmp_issues[grep('general_bugs', tmp_issues$project_type),]
perf_bugs = tmp_issues[grep('perf_bugs', tmp_issues$project_type),]

par(mar = c(5.5, 4, 4, 2), xpd=TRUE)
beanplot(general_bugs$resolve_time, perf_bugs$resolve_time, ll = 0.04, what = c(0,1,1,0), xlab="", cex.lab=1.5, cex.axis = 1.5, yaxt="n",axes=T,xaxt="n", beanlines = 'median',
         ylab = "Hours taken to resolve bugs after assigning", side = "both", col = list("black", c("grey", "white")), bw=1.3, log = 'y')
axis(2, at = c(0.01, 120, 1e+06), labels = c("0.01", "120 (i.e., 5 days)",  "1e+06"), cex.axis = 1.3)
legend("bottomleft",  inset=c(0,-0.2), fill = c("black", "grey"),
           legend = c("General bugs", "Performance bugs"), cex=1.5, horiz=T)

median(general_bugs$resolve_time)
median(perf_bugs$resolve_time)
wilcox.test(general_bugs$resolve_time, perf_bugs$resolve_time)
cliff.delta(general_bugs$resolve_time, perf_bugs$resolve_time)


tmp_issues = issues[issues$assign_time > 0, ]
general_bugs = tmp_issues[grep('general_bugs', tmp_issues$project_type),]
perf_bugs = tmp_issues[grep('perf_bugs', tmp_issues$project_type),]


par(mar = c(5.5, 4, 4, 2), xpd=TRUE)
beanplot(general_bugs$assign_time, perf_bugs$assign_time,  ll = 0.04, what = c(0,1,1,0), xlab="", cex.lab=1.5, cex.axis = 1.5, beanlines="median", yaxt="n",axes=T,xaxt="n",
         ylab = "Hours taken to assign bugs", side = "both", col = list("black", c("grey", "white")),log='y', bw=1.3)
axis(2, at = c(0.01, 120, 1e+06), labels = c("0.01", "120 (i.e., 5 days)",  "1e+06"),cex.axis = 1.5)
legend("bottomleft",  inset=c(0,-0.2), fill = c("black", "grey"),
       legend = c("General bugs", "Performance bugs"), cex=1.3, horiz=T)

general_bugs = issues[grep('general_bugs', issues$project_type),]
perf_bugs = issues[grep('perf_bugs', issues$project_type),]
median(general_bugs$assign_time)
median(perf_bugs$assign_time)
wilcox.test(general_bugs$assign_time, perf_bugs$assign_time)
cliff.delta(general_bugs$assign_time, perf_bugs$assign_time)


tmp_issues = issues[issues$resolve_time > 0, ]
general_bugs = tmp_issues[grep('general_bugs', tmp_issues$project_type),]
perf_bugs = tmp_issues[grep('perf_bugs', tmp_issues$project_type),]


par(mar = c(5.5, 4, 4, 2), xpd=TRUE)
beanplot(general_bugs$num_developers, perf_bugs$num_developers, ll = 0.04, what = c(0,1,1,0), xlab="", cex.lab=1.5, beanlines="median",cex.axis = 1.5,axes=T,xaxt="n",
         ylab = "Number of developers in addressing bugs", side = "both", col = list("black", c("grey", "white")), bw=0.55)
legend("bottomleft",  inset=c(0,-0.2), fill = c("black", "grey"),
       legend = c("General bugs", "Performance bugs"), cex=1.5, horiz=T)


median(general_bugs$num_developers)
median(perf_bugs$num_developers)
wilcox.test(general_bugs$num_developers, perf_bugs$num_developers)
cliff.delta(general_bugs$num_developers, perf_bugs$num_developers)


tmp_issues = issues[issues$resolve_time > 0, ]
general_bugs = tmp_issues[grep('general_bugs', tmp_issues$project_type),]
perf_bugs = tmp_issues[grep('perf_bugs', tmp_issues$project_type),]


par(mar = c(5.5, 4, 4, 2), xpd=TRUE)
beanplot(general_bugs$num_comments, perf_bugs$num_comments, ll = 0.04, what = c(0,1,1,0), xlab="", cex.lab=1.5, cex.axis = 1.5, beanlines="median", axes=T,xaxt="n",
         ylab = "Number of comments from developers", side = "both", col = list("black", c("grey", "white")), bw=0.6)
legend("bottomleft",  inset=c(0,-0.2), fill = c("black", "grey"),
       legend = c("General bugs", "Performance bugs"), cex=1.5, horiz=T)


median(general_bugs$num_comments)
median(perf_bugs$num_comments)
print(wilcox.test(general_bugs$num_comments, perf_bugs$num_comments))
print(cliff.delta(general_bugs$num_comments, perf_bugs$num_comments))
