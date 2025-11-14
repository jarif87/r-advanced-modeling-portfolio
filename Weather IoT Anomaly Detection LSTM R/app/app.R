# ========================================
# app.R - FINAL SHINY DASHBOARD (R STUDIO)
# ========================================
library(shiny)
library(ggplot2)
library(dplyr)
library(DT)
library(lubridate)

# --- LOAD DATA ---
submission   <- read.csv("submission.csv")
df_raw       <- read.csv("data_with_time.csv")
model_stats  <- read.csv("model_stats.csv")  # FIXED: Load threshold

# --- CREATE DATETIME SAFELY ---
SEQ_LEN <- 60

df_plot <- df_raw %>%
  slice((SEQ_LEN + 1):n()) %>%
  mutate(
    datetime = ymd_h(paste(year, month, day, hour), quiet = TRUE),
    datetime = if_else(is.na(datetime),
                       as.POSIXct("2020-01-01") + hours(row_number()),
                       datetime),
    date = as.Date(datetime),
    is_anomaly = submission$is_anomaly
  ) %>%
  select(date, datetime, year, month, day, hour,
         Tpot, Tdew, rh, VPmax, VPact, VPdef, sh,
         H2OC, rho, wv, `max..wv`, wd, rain, raining,
         SWDR, PAR, `max..PAR`, Tlog, is_anomaly)

# --- UI ---
ui <- fluidPage(
  titlePanel("IoT Anomaly Detection - LSTM Autoencoder"),
  sidebarLayout(
    sidebarPanel(
      selectInput("sensor", "Select Sensor:",
                  choices = names(df_plot)[7:24], selected = "Tpot"),
      dateRangeInput("date_range", "Date Range:",
                     start = min(df_plot$date, na.rm = TRUE),
                     end = max(df_plot$date, na.rm = TRUE),
                     format = "yyyy-mm-dd"),
      h4("Model Summary"),
      verbatimTextOutput("summary")
    ),
    mainPanel(
      plotOutput("plot", height = "600px"),
      DTOutput("table")
    )
  )
)

# --- SERVER ---
server <- function(input, output) {
  
  filtered <- reactive({
    df_plot %>%
      filter(date >= input$date_range[1],
             date <= input$date_range[2])
  })
  
  output$plot <- renderPlot({
    ggplot(filtered(), aes(x = datetime, y = .data[[input$sensor]])) +
      geom_line(color = "steelblue", size = 0.8) +
      geom_point(data = filter(filtered(), is_anomaly == 1),
                 color = "red", size = 2.5, alpha = 0.8) +
      scale_x_datetime(date_breaks = "1 day", date_labels = "%b %d") +
      labs(title = paste("Sensor:", input$sensor, "| Red = Anomaly"),
           x = "Date", y = input$sensor) +
      theme_minimal(base_size = 14) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  output$table <- renderDT({
    filtered() %>%
      filter(is_anomaly == 1) %>%
      select(datetime, all_of(input$sensor)) %>%
      mutate(datetime = format(datetime, "%Y-%m-%d %H:%M")) %>%
      rename(Time = datetime, Value = all_of(input$sensor)) %>%
      datatable(options = list(pageLength = 8, scrollX = TRUE))
  })
  
  # FIXED: Use model_stats$threshold
  output$summary <- renderText({
    paste0(
      "Total Points: ", model_stats$total_points, "\n",
      "Anomalies: ", model_stats$anomalies, 
      " (", round(100 * model_stats$anomaly_rate, 2), "%)\n",
      "Threshold: ", round(model_stats$threshold, 4), "\n",
      "Model: LSTM Autoencoder (R + Keras)"
    )
  })
}

# --- RUN ---
shinyApp(ui, server)