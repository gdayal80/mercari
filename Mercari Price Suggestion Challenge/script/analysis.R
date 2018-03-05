install.packages("ggplot2") # Data visualization
install.packages("treemapify") # Treemap visualization
install.packages("gridExtra") # Create multiplot
install.packages("dplyr") # data manipulation
install.packages("tidyr") # data manipulation
install.packages("tibble") # data wrangling
install.packages("stringr") # String processing
install.packages("repr")
install.packages("stringi") # String processing
install.packages("data.table") # Loading data

library(data.table) # Loading data
library(ggplot2) # Data visualization
library(treemapify) # Treemap visualization
library(gridExtra) # Create multiplot
library(dplyr) # data manipulation
library(tidyr) # data manipulation
library(tibble) # data wrangling
library(stringr) # String processing
library(repr)
library(stringi) # String processing

setwd("C:/Users/Gautam/Downloads/Data Science/Mercari Price Suggestion Challenge/script/")
getwd()

train = fread('../input/train.tsv', sep='\t')
test = fread('../input/test.tsv', sep='\t')

summary(train)
summary(test)

train = train %>% mutate(log_price = log(price+1)) # take log of the price
train = train %>% mutate(item_condition_id = factor(item_condition_id))
train = train %>% mutate(shipping = factor(shipping))

options(repr.plot.width=7, repr.plot.height=7)

p1 = train %>% ggplot(aes(x=log_price)) +
  geom_histogram(bins=30) +
  ggtitle('Distributon of Log1p Price')

p2 = train %>% ggplot(aes(x=price)) +
  geom_histogram(bins=30) +
  xlim(0,300) +
  ggtitle('Distributon of Price')

p3 = train %>% ggplot(aes(x=item_condition_id)) +
  geom_bar() +
  ggtitle('Distribution of Item Conditions') +
  theme(legend.position="none")

p4 = train %>% ggplot(aes(x=shipping)) +
  geom_bar(width=0.5) +
  ggtitle('Distribution of Shipping Info') +
  theme(legend.position="none")

suppressWarnings(grid.arrange(p1, p2, p3, p4, ncol=2))

train = data.frame(train, str_split_fixed(train$category_name, '/', 4)) %>%
  mutate(cat1=X1, cat2=X2, cat3=X3, cat4=X4) %>% select(-X1, -X2, -X3, -X4)

train %>% summarise(Num_Cat1 = length(unique(cat1)), Num_Cat2 = length(unique(cat2)),
                    Num_Cat3 = length(unique(cat3)), Num_Cat4 = length(unique(cat4)))


options(repr.plot.width=7, repr.plot.height=7)

train %>%
  group_by(cat1, cat2) %>%
  count() %>%
  ungroup() %>%
  ggplot(aes(area=n, fill=cat1, label=cat2, subgroup=cat1)) +
  geom_treemap() +
  geom_treemap_subgroup_text(grow = T, alpha = 0.5, colour =
                               "black", fontface = "italic", min.size = 0) +
  geom_treemap_text(colour = "white", place = "topleft", reflow = T) +
  theme(legend.position = "null") +
  ggtitle("1st and 2nd Hierarchical Category Levels")


options(repr.plot.width=7, repr.plot.height=7)

p1 = train %>% count(cat1) %>% 
  ggplot(aes(x=reorder(cat1, -n), y=n)) +
  geom_bar(stat='identity', width=0.7) +
  ggtitle('1st Level Categories') +
  xlab('1st Level Catogory') +
  ylab('count') +
  theme(axis.text.x=element_text(angle=15, hjust=1))

p2 = train %>% count(cat2) %>% 
  filter(n>20000) %>% 
  ggplot(aes(x=reorder(cat2,-n), y=n)) +
  geom_bar(stat='identity', width=0.7) +
  ggtitle('2nd Level Categories (>20000 only)') +
  xlab('2nd Level Catogory') +
  ylab('count') +
  theme(axis.text.x=element_text(angle=15, hjust=1, size=7))

grid.arrange(p1, p2, ncol=1)

options(repr.plot.width=7, repr.plot.height=7)

train %>% filter(cat1=='Women') %>% 
  group_by(cat2, cat3) %>%
  count() %>%
  ungroup() %>%
  ggplot(aes(area=n, fill=cat2, label=cat3, subgroup=cat2)) +
  geom_treemap() +
  geom_treemap_subgroup_text(grow = T, alpha = 0.5, colour =
                               "black", fontface = "italic", min.size = 0) +
  geom_treemap_text(colour = "white", place = "topleft", reflow = T) +
  theme(legend.position = "null") +
  ggtitle("2nd and 3rd Hierarchical Category Levels Under Woman")


options(repr.plot.width=7, repr.plot.height=3.5)

train %>% filter(cat1=='Women') %>%
  count(cat2) %>% 
  ggplot(aes(x=reorder(cat2, -n), y=n)) +
  geom_bar(stat='identity', width=0.7) +
  ggtitle('2nd Level Categories Under Women') +
  xlab('2nd Level Catogory Under Women') +
  ylab('count') +
  theme(axis.text.x=element_text(angle=30, hjust=1, size=8))

options(repr.plot.width=7, repr.plot.height=3.5)

train = train %>% mutate(has_brand=(brand_name!=''))
train %>%
  ggplot(aes(x=cat1, fill=has_brand)) +
  geom_bar(position='fill') +
  theme(axis.text.x=element_text(angle=15, hjust=1, size=8)) +
  xlab('1st Categories') +
  ylab('Proportion') +
  ggtitle('Items With and Without Brands')

options(repr.plot.width=7, repr.plot.height=3.5)

top10 = train %>% filter(brand_name!='') %>% 
  count(brand_name) %>%
  arrange(desc(n)) %>%
  head(10)

train %>% filter(brand_name %in% top10$brand_name) %>%
  ggplot(aes(x=factor(brand_name, levels=rev(top10$brand_name)), fill=cat1)) +
  geom_bar(width=0.5) +
  coord_flip() +
  xlab('brand') +
  labs(fill='1st Category') +
  ggtitle('Top 10 Brands and Their Categories')


options(repr.plot.width=7, repr.plot.height=3.5)

p1 = train %>% mutate(len_of_des = str_length(item_description)) %>%
  ggplot(aes(x=len_of_des)) +
  geom_histogram(bins=50) +
  ggtitle('Distribution of Length of Descriptions') +
  xlab('Length of Item Description') +
  theme(plot.title = element_text(size=10))

p2 = train %>% mutate(num_token_des = str_count(item_description, '\\S+')) %>% 
  ggplot(aes(x=num_token_des)) +
  geom_histogram(bins=50) +
  ggtitle('Distribution of # of Tokens of Descriptions') +
  xlab('Number of Tokens') +
  theme(plot.title = element_text(size=10))

grid.arrange(p1, p2, ncol=2)

options(repr.plot.width=7, repr.plot.height=3.5)

train = train %>% mutate(num_token_name = str_count(name, '\\S+'))
train %>%
  ggplot(aes(x=num_token_name)) +
  geom_bar(width=0.7) +
  ggtitle('Distribution of # of Tokens of Names') +
  xlab('Number of Words')

options(repr.plot.width=7, repr.plot.height=7)

p1 = train %>% filter(cat1!='') %>% 
  mutate(cat3 = as.character(cat3)) %>% 
  mutate(cat_in_name = (str_detect(name, cat3))) %>% 
  ggplot(aes(x=cat1, fill=cat_in_name)) +
  geom_bar(position='fill') +
  theme(axis.text.x=element_text(angle=30, hjust=1, size=8)) +
  xlab('1st Categories') +
  ylab('Proportion') +
  ggtitle('3rd Category Appears in Item Name')

p2 = train %>% filter(has_brand) %>% 
  mutate(brand_name = as.character(brand_name)) %>% 
  mutate(brand_in_name = (str_detect(name, brand_name))) %>% 
  ggplot(aes(x=cat1, fill=brand_in_name)) +
  geom_bar(position='fill') +
  theme(axis.text.x=element_text(angle=30, hjust=1, size=8)) +
  xlab('1st Categories') +
  ylab('Proportion') +
  ggtitle('Brand Appears in Item Name')

grid.arrange(p1, p2, ncol=1)

options(repr.plot.width=7, repr.plot.height=3.5)

train %>% filter(brand_name %in% top10$brand_name) %>%
  mutate(brand_name = as.character(brand_name)) %>% 
  mutate(brand_in_name = (str_detect(name, brand_name))) %>% 
  ggplot(aes(x=factor(brand_name, levels=top10$brand_name), fill=brand_in_name)) +
  geom_bar(position='fill') +
  theme(axis.text.x=element_text(angle=30, hjust=1, size=8)) +
  xlab('1st Categories') +
  ylab('Proportion') +
  ggtitle('Brand Appears in Item Name (By Top 10 Brands)')

options(repr.plot.width=7, repr.plot.height=3.5)

p1 = train %>%
  ggplot(aes(x=item_condition_id, y=log_price, fill=item_condition_id)) +
  geom_boxplot(outlier.size=0.1) +
  ggtitle('Boxplot of Log Price versus Condition') +
  theme(legend.position="none", plot.title = element_text(size=10))

p2 = train %>%
  ggplot(aes(x=shipping, y=log_price, fill=shipping)) +
  geom_boxplot(width=0.5, outlier.size=0.1) +
  ggtitle('Boxplot of Log Price versus Shipping') +
  theme(legend.position="none", plot.title = element_text(size=10))

grid.arrange(p1, p2, ncol=2)

options(repr.plot.width=7, repr.plot.height=3.5)

train %>%
  ggplot(aes(x=cat1, y=log_price, fill=has_brand)) +
  geom_boxplot(outlier.size=0.1) +
  ggtitle('Boxplot of Log Price versus 1st Category') +
  xlab('1st Category') +
  theme(axis.text.x=element_text(angle=15, hjust=1))

options(repr.plot.width=7, repr.plot.height=7)

train %>% mutate(len_of_des = str_length(item_description)) %>%
  group_by(len_of_des) %>%
  summarise(mean_log_price = mean(log_price)) %>% 
  ggplot(aes(x=len_of_des, y=mean_log_price)) +
  geom_point(size=0.5) +
  geom_smooth(method = "loess", color = "red", size=0.5) +
  ggtitle('Mean Log Price versus Length of Description')

options(repr.plot.width=7, repr.plot.height=7)

train %>% mutate(num_token_des = str_count(item_description, '\\S+')) %>%
  group_by(num_token_des) %>%
  summarise(mean_log_price = mean(log_price)) %>% 
  ggplot(aes(x=num_token_des, y=mean_log_price)) +
  geom_point(size=0.5) +
  geom_smooth(method = "loess", color = "red", size=0.5) +
  ggtitle('Mean Log Price versus # of Tokens of Description')

