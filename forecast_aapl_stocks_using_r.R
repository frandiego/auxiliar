
# FORECAST AAPL STOCKS PRICE USING R  ------------------------------------

# DEPENDENCIES -------------------
# install.packages(c('data.table',
#                    'magrittr',
#                    'prophet',
#                    'mFilter',
#                    'dygraphs',
#                    'lubridate',
#                    'xts',
#                    'ggplot2'))

library(data.table)
library(magrittr)
library(prophet)
library(mFilter)
library(dygraphs)
library(lubridate)
library(xts)
library(ggplot2)



# READ DATA ---------------------------------------------------------------
path <- 'https://raw.githubusercontent.com/frandiego/data/master/aapl_stocks.csv'
df <- fread(path)
str(df)

# TIDY --------------------------------------------------------------------
df <- df[,c('date','adjOpen','adjHigh','adjLow','adjClose'),with=F]
df[,date := as.Date(date)]
str(df)

# VISUALIZE ----------------------------------------------------------------
df %>% 
  as.xts.data.table() %>% 
  dygraph()

df %>% 
  as.xts.data.table() %>% 
  dygraph() %>% 
  dyCandlestick() %>% 
  dyRangeSelector()

# PREPROCESS ---------------------------------------------------------------
df <- df[,c('date','adjClose'),with=F]
colnames(df) <- c('ds','y')
df <- unique(df)
head(df)
tail(df)

# TRAIN AND TEST SPLIT ---------------------------------------------------
th_date <- df[,max(ds)] - lubridate::years(1)
train <- df[ds<th_date]
test <- df[ds>=th_date]
train[,ds := as.POSIXct.Date(ds)]
test[,ds := as.POSIXct.Date(ds)]

# number of days to predict
print(test[,max(ds)] - max(train$ds))

# FIT A PROPHET MODEL -----------------------------------------------------
model <- prophet(train,changepoint.range = 1,
                 changepoint.prior.scale = 100)

# CREATE THE FUTURE DATASET ------------------------------------------------
diff_days <- as.integer(test[,max(ds)] - max(train$ds))
future <- make_future_dataframe(model, 
                                periods = diff_days,
                                freq = 'day')
head(future)
tail(future)

# CREATE THE FUTURE DATASET ------------------------------------------------
future <- make_future_dataframe(model, 
                                periods = diff_days,
                                freq = 'day')
head(future)
tail(future)

# PREDICT ------------------------------------------------------------------
forecast <- predict(model, future)
head(forecast)

# VISUAL EVALUATION OF OUR FORECAST ----------------------------------------
plot(model, forecast) + 
  add_changepoints_to_plot(model)+
  geom_point(data = test,aes(x=ds,y=y),color='red')+
  theme_minimal(base_size =20)+
  labs(x=element_blank(),y='adjusted close price', 
       title = 'AAPL Stock Prediction')


# USE SOME ECONOMICS TRICKS -------------------------------------------------
# https://www.stata.com/manuals13/tstsfilterhp.pdf
hodrick_prescott <- hpfilter(train$y, type='lambda',
                             freq= 1600 * (365/4) )
# plot the hodrick_prescott
plot(hodrick_prescott)

#  plot the hodrick_prescott trend
hodrick_prescott$trend %>% plot(type='l')

#  plot the growth rate of hodrick_prescott trend
hodrick_prescott$trend %>% log %>% diff %>%  plot(type='l')

# find the change points dates
data.table(c(0,hodrick_prescott$trend %>% log %>% diff)) %>% 
  cbind(train) %>% 
  setnames(c('gr','ds','y')) %>% 
  setcolorder(c('ds','y','gr')) -> df_dates

upper_quantile = df_dates$gr %>% quantile(0.9)
lower_quantile = df_dates$gr %>% quantile(0.1)

ggplot(df_dates,aes(x=ds,y=gr))+
  geom_line(size=1)+
  geom_hline(yintercept = upper_quantile ,color='red')+
  geom_hline(yintercept = lower_quantile ,color='red')+
  theme_minimal()

# compute changepoints using the trend of hodrick_prescott
grow_rate  <- hodrick_prescott$trend %>% log %>% diff %>% abs
grow_rate <- c(0,grow_rate)
max_grow_rate <- as.numeric(quantile(grow_rate,0.9))

# selecting the dates
dates <- train$ds[which(grow_rate>=max_grow_rate)]
dates

# TRAIN NEW ML-ECONOMICS MODEL  -------------------------------------------------

# train the model taking into account the structural changes dates
model <- prophet(train,changepoint.range = 1,
                 changepoint.prior.scale = 100,
                 changepoints = dates)

# PREDICT WITH NEW ML-ECONOMICS MODEL  -------------------------------------------------
# create the future data frame and predict
future <- make_future_dataframe(model, 
                                periods = diff_days,
                                freq = 'day')
forecast <- predict(model, future)

# VISUALIZE THE OUTPUT OF THE NEW ML-ECONOMICS MODEL  ------------------------------

# plot the new model
plot(model, forecast) + 
  add_changepoints_to_plot(model)+
  geom_point(data = test,aes(x=ds,y=y),color='red')+
  theme_minimal(base_size =20)+
  labs(x=element_blank(),y='adjusted close price', 
       title = 'AAPL Stock Prediction')

# plot the seasonal compoennts of the model
prophet_plot_components(model, forecast)