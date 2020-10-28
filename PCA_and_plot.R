## This is a R script for generating principal components analysis (PCA) and visualisation.

## Written and debugged by Arthur Zhu on 28/10/2020


### Import packages ###
## ============================================================ ##
library('readxl') # for importing data from Excel files
library('ggord') # dev version from fawda123/ggord")
library('class')
library('ggpubr') # dev version from kassambara/ggpubr
library('cowplot') # dev version from wilkelab/cowplot
library('pheatmap')



### Read data from the Excel file ###
## ============================================================ ##
all_data <- read_excel('Tanks2.xlsx', sheet = 'Pcutoff')
peaks <- all_data %>% dplyr::select(15:79)
peaks <- normalize(peaks)
peaks <- data.frame(sapply(peaks, function(x){x*1000})) # Adjust normalisation scale to 1-1000
with_sample_names <- cbind(all_data$Vessel_Out, all_data$Grade, peaks)
colnames(with_sample_names)[1] <- 'Sample'
colnames(with_sample_names)[2] <- 'Grade'
row.names(with_sample_names) <- with_sample_names$Sample
with_sample_names$Sample <- NULL
pca_ready <- with_sample_names



### Calculate PCA ###
## ============================================================ ##
wine_pca <- prcomp(pca_ready[, 2:66], scale=FALSE)
wine_pca_score <- data.frame(Grade=pca_ready$Grade,
                             PC1=wine_pca$x[,1],
                             PC2=wine_pca$x[,2], 
                             PC3=wine_pca$x[,3],
                             PC4=wine_pca$x[,4],
                             PC5=wine_pca$x[,5],
                             PC6=wine_pca$x[,6],
                             PC7=wine_pca$x[,7],
                             PC8=wine_pca$x[,8],
                             PC9=wine_pca$x[,9],
                             PC10=wine_pca$x[,10])



### Visualise PCA ###
## ============================================================ ##
not_fancy_pca_plot <- ggord(wine_pca, arrow=0, grp_in=wine_pca_score$Grade, grp_title = 'Grade', # grouping factor as grade
                            txt=NULL, size=2, vec_ext=0,
                            col=pal_nejm('default')(3), # colour palatte
                            ellipse=TRUE, ellipse_pro=0.95, alpha_el=0.15) + 
                      theme_linedraw() +
                      coord_fixed(1) +
                      theme(text = element_text(size=14, family='Open Sans'),
                            panel.border = element_rect(size=0.7),
                            panel.grid.major = element_line(colour="gray"),
                            panel.grid.minor = element_blank(),
                            panel.background = element_blank(),
                            legend.position = 'right',
                            legend.box.margin=margin(l=0, unit='pt'),
                            axis.title.x = element_text(margin=margin(t=8, unit='pt')),
                            axis.title.y = element_text(margin=margin(r=8, unit='pt')))
# ------------------------------------------------------------------
# marginal histogram on PC1
xhist <- axis_canvas(not_fancy_pca_plot, axis = "x") +
    geom_histogram(data = wine_pca_score, position = 'identity', binwidth = 100,
                   aes(x = PC1, fill = Grade),
                   alpha = 0.25, size = 2) +
    scale_fill_manual(values = pal_nejm('default')(3))
# ------------------------------------------------------------------
# marginal histogram on PC2
yhist <- axis_canvas(not_fancy_pca_plot, axis = "y", coord_flip = TRUE) +
    geom_histogram(data = wine_pca_score, position = 'identity', binwidth = 100,
                   aes(x = PC2, fill = Grade),
                   alpha = 0.25, size = 2) +
    coord_flip() +
    scale_fill_manual(values = pal_nejm('default')(3))
# ------------------------------------------------------------------
# glue everything together and output the plot
lightly_fancier_pca_plot <- insert_xaxis_grob(not_fancy_pca_plot, xhist, grid::unit(.2, "null"), position = "top")
very_fancy_pca_plot <- insert_yaxis_grob(slightly_fancier_pca_plot, yhist, grid::unit(.2, "null"), position = "right")
ggdraw(very_fancy_pca_plot)
