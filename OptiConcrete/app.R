library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(data.table)
library(ggplot2)

archive <- readRDS("concrete_pareto_archive.rds")

ui <- dashboardPage(
  dashboardHeader(title = "Concrete Strength: Accuracy vs Speed"),
  dashboardSidebar(disable = TRUE),
  dashboardBody(
    fluidRow(
      box(title = "Pareto Front Evolution", width = 8, solidHeader = TRUE, status = "primary",
          sliderInput("step", "Optimization Step:", 10, nrow(archive), nrow(archive), step = 1, animate = animationOptions(100)),
          plotlyOutput("pareto_evolution", height = "600px")),
      box(title = "Best Solutions", width = 4, solidHeader = TRUE, status = "warning",
          DTOutput("pareto_table"))
    ),
    fluidRow(
      box(title = "Final Selected Model", width = 12, status = "success",
          verbatimTextOutput("final_perf"))
    )
  )
)

server <- function(input, output) {
  current <- reactive({ archive[1:input$step] })
  
  pareto_current <- reactive({ 
    curr <- current()
    setorder(curr, rmse, pred_time_us)
    curr[, is_pareto_temp := TRUE]
    best_time <- Inf
    for (i in seq_len(nrow(curr))) {
      if (curr[i, pred_time_us] >= best_time) {
        curr[i, is_pareto_temp := FALSE]
      } else {
        best_time <- curr[i, pred_time_us]
      }
    }
    curr[is_pareto_temp == TRUE]
  })
  
  output$pareto_evolution <- renderPlotly({
    p <- ggplot(current(), aes(x = pred_time_us, y = rmse)) +
      geom_point(color = "gray", alpha = 0.7) +
      geom_point(data = pareto_current(), color = "#e31a1c", size = 4) +
      geom_line(data = pareto_current(), color = "#e31a1c", size = 1.2) +
      theme_minimal() + 
      labs(x = "Prediction Time (μs)", y = "RMSE (MPa)",
           title = paste("Step", input$step, "– Pareto Front")) +
      xlim(range(archive$pred_time_us)) + ylim(range(archive$rmse))
    ggplotly(p)
  })
  
  output$pareto_table <- renderDT({
    pareto_current()[order(rmse)][1:min(10, .N),
                                  .(RMSE = round(rmse, 3), 
                                    `Time (μs)` = round(pred_time_us, 1),
                                    Depth = max_depth, 
                                    ETA = round(eta, 3), 
                                    Subsample = round(subsample, 2))]
  }, options = list(pageLength = 10))
  
  output$final_perf <- renderPrint({
    best <- archive[which.min(rmse + pred_time_us/1000)]
    cat("Best Compromise Model:\n")
    cat(sprintf("RMSE: %.3f MPa\nPrediction Time: %.1f μs\n", best$rmse, best$pred_time_us))
  })
}

shinyApp(ui, server)