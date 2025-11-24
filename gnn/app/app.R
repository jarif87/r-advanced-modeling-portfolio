library(shiny)
library(ggraph)
library(tidygraph)
library(shinythemes)
library(DT)
library(igraph)
library(ggplot2)
library(torch)
library(dplyr)
library(viridis)

if (!file.exists("gnn_influencer_app_data.RData")) 
  stop("Missing gnn_influencer_app_data.RData - Please run the save command first!")
if (!file.exists("gnn_influencer_model_state.pt")) 
  stop("Missing gnn_influencer_model_state.pt - Please run the save command first!")

cat("Loading data...\n")
load("gnn_influencer_app_data.RData")

cat("Rebuilding tensors from saved arrays...\n")
data <- list(
  x = torch_tensor(save_data$x, dtype = torch_float()),
  edge_index = torch_tensor(save_data$edge_index, dtype = torch_long()),
  y = torch_tensor(save_data$y, dtype = torch_long()),
  test_mask = torch_tensor(save_data$test_mask, dtype = torch_bool()),
  num_nodes = save_data$num_nodes
)

cat("Tensors rebuilt successfully!\n")

mean_agg <- function(x, edge_index, num_nodes) {
  src <- edge_index[1, ]
  dst <- edge_index[2, ]
  out <- torch_zeros(num_nodes, x$size(2), dtype = x$dtype, device = x$device)
  out$index_add_(1, dst, x[src, ])
  deg <- torch_zeros(num_nodes, 1, dtype = x$dtype, device = x$device)
  deg$index_add_(1, dst, torch_ones(src$size(1), 1, dtype = x$dtype, device = x$device))
  out / deg$clamp(min = 1)
}

simple_layer <- nn_module(
  initialize = function(in_dim, out_dim) {
    self$lin <- nn_linear(in_dim, out_dim)
    self$norm <- nn_layer_norm(out_dim)
    self$res_proj <- if (in_dim != out_dim) nn_linear(in_dim, out_dim, bias = FALSE) else NULL
  },
  forward = function(x, edge_index, num_nodes) {
    h <- mean_agg(x, edge_index, num_nodes)
    h <- self$lin(h)
    h <- nnf_gelu(h)
    h <- self$norm(h)
    if (!is.null(self$res_proj)) x <- self$res_proj(x)
    h + x
  }
)

model <- nn_module(
  initialize = function(input_dim) {
    self$proj  <- nn_linear(input_dim, 512)
    self$norm0 <- nn_layer_norm(512)
    self$l1 <- simple_layer(512, 512)
    self$l2 <- simple_layer(512, 512)
    self$l3 <- simple_layer(512, 256)
    self$l4 <- simple_layer(256, 256)
    self$l5 <- simple_layer(256, 128)
    self$clf <- nn_sequential(
      nn_linear(128, 256), nn_gelu(), nn_dropout(0.5),
      nn_linear(256, 128), nn_gelu(), nn_dropout(0.5),
      nn_linear(128, 2)
    )
  },
  forward = function(x, edge_index) {
    h <- self$proj(x)
    h <- nnf_gelu(self$norm0(h))
    n <- x$size(1)
    h <- self$l1(h, edge_index, n)
    h <- self$l2(h, edge_index, n)
    h <- self$l3(h, edge_index, n)
    h <- self$l4(h, edge_index, n)
    h <- self$l5(h, edge_index, n)
    self$clf(h)
  }
)

cat("Loading model...\n")
net <- model(data$x$size(2))
net$load_state_dict(torch_load("gnn_influencer_model_state.pt"))
net$eval()

cat("Running inference...\n")
with_no_grad({
  out <- net(data$x, data$edge_index)
  prob <- out$softmax(dim = 2)
  prob_cpu <- as.array(prob$cpu())
  pred <- out$argmax(dim = 2) + 1L
})

V(g_hr)$score <- prob_cpu[, 2]
V(g_hr)$label <- ifelse(prob_cpu[, 2] > 0.5, "Influential", "Regular")
V(g_hr)$true_label <- ifelse(V(g_hr)$influential == 1, "True Influencer", "Not Influencer")

h <- as.array(out$cpu())
pca <- prcomp(h, center = TRUE, scale. = TRUE)
embed_df <- data.frame(
  PC1 = pca$x[,1], 
  PC2 = pca$x[,2],
  score = V(g_hr)$score,
  degree = log1p(degree(g_hr))
)

test_idx <- which(save_data$test_mask)
y_true <- save_data$y[test_idx]
y_pred <- as.array(pred$cpu())[test_idx]
prob_test <- prob_cpu[test_idx, 2]

acc <- round(mean(y_pred == y_true) * 100, 1)
rec <- if(sum(y_true == 2) > 0) {
  round(sum((y_pred == 2) & (y_true == 2)) / sum(y_true == 2) * 100, 1)
} else { 0 }
pre <- if(sum(y_pred == 2) > 0) {
  round(sum((y_pred == 2) & (y_true == 2)) / sum(y_pred == 2) * 100, 1)
} else { 0 }

cat("App ready to launch!\n")
cat(sprintf("Accuracy: %.1f%% | Precision: %.1f%% | Recall: %.1f%%\n", acc, pre, rec))

ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("🎵 GNN Influencer Detection – Deezer Croatia"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("🎛️ Controls"),
      sliderInput("topn", "Show top N influencers:", 5, 30, 10),
      selectInput("color", "Color nodes by:", c("GNN Score", "Degree", "True Label")),
      actionButton("refresh", "🔄 Refresh", icon("sync"), class = "btn-primary btn-block"),
      tags$hr(),
      h4("📊 Model Info"),
      tags$p(strong("Architecture:"), "5-layer GraphSAGE"),
      tags$p(strong("Features:"), "Genre + Degree"),
      tags$p(strong("Accuracy:"), paste0(acc, "%")),
      tags$p(strong("Precision:"), paste0(pre, "%")),
      tags$p(strong("Recall:"), paste0(rec, "%")),
      tags$hr(),
      tags$p(
        style = "font-size: 11px; color: #7f8c8d;",
        "Graph Neural Network trained on music preferences and social connections."
      )
    ),
    mainPanel(
      width = 9,
      tabsetPanel(
        tabPanel("🌐 Network", 
                 br(),
                 plotOutput("graph", height = "700px")),
        tabPanel("📋 Top 20", 
                 br(),
                 DTOutput("table")),
        tabPanel("🗺️ Embeddings", 
                 br(),
                 plotOutput("emb", height = "600px")),
        tabPanel("📊 Analysis", 
                 br(),
                 verbatimTextOutput("metrics"),
                 plotOutput("prob_hist", height = "400px"))
      )
    )
  )
)

server <- function(input, output) {
  
  subgraph <- reactive({
    input$refresh
    top <- order(V(g_hr)$score, decreasing = TRUE)[1:input$topn]
    neigh <- unique(unlist(neighborhood(g_hr, 1, top)))
    induced_subgraph(g_hr, neigh)
  })
  
  output$graph <- renderPlot({
    g <- subgraph()
    
    if (input$color == "GNN Score") {
      col <- V(g)$score
      scale_col <- scale_color_viridis_c(option = "plasma", name = "GNN Score")
    } else if (input$color == "Degree") {
      col <- degree(g)
      scale_col <- scale_color_viridis_c(option = "viridis", name = "Degree")
    } else {
      col <- V(g)$true_label
      scale_col <- scale_color_manual(
        values = c("True Influencer" = "#d95f0e", "Not Influencer" = "#2166ac"),
        name = "True Label"
      )
    }
    
    ggraph(g, layout = "fr") +
      geom_edge_link(alpha = 0.2, color = "gray60") +
      geom_node_point(aes(color = col, size = degree(g)), alpha = 0.9) +
      scale_col + 
      scale_size_continuous(range = c(3, 20), name = "Degree") +
      labs(title = paste("Top", input$topn, "Predicted Influencers & Network")) +
      theme_void(base_size = 14) + 
      theme(legend.position = "bottom",
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5))
  })
  
  output$table <- renderDT({
    idx <- order(V(g_hr)$score, decreasing = TRUE)[1:20]
    data.frame(
      Rank = 1:20,
      User = V(g_hr)$name[idx],
      Degree = degree(g_hr)[idx],
      Score = round(V(g_hr)$score[idx], 4),
      Prediction = V(g_hr)$label[idx],
      Truth = ifelse(V(g_hr)$influential[idx] == 1, "✓ Influential", "✗ Regular")
    ) %>% 
      datatable(
        rownames = FALSE, 
        options = list(pageLength = 10, searching = FALSE, dom = 'tip'),
        class = 'cell-border stripe'
      ) %>%
      formatStyle(
        'Score',
        background = styleColorBar(c(0, 1), '#fee5d9'),
        backgroundSize = '100% 90%',
        backgroundRepeat = 'no-repeat',
        backgroundPosition = 'center'
      )
  })
  
  output$emb <- renderPlot({
    ggplot(embed_df, aes(PC1, PC2, color = score, size = degree)) +
      geom_point(alpha = 0.7) +
      scale_color_viridis_c(option = "plasma", name = "GNN Score") +
      scale_size_continuous(range = c(1, 6), name = "Log Degree") +
      labs(title = "Node Embeddings in 2D Space (PCA)",
           subtitle = "Nodes with similar learned features cluster together",
           x = paste0("PC1 (", round(summary(pca)$importance[2,1]*100, 1), "% variance)"),
           y = paste0("PC2 (", round(summary(pca)$importance[2,2]*100, 1), "% variance)")) +
      theme_minimal(base_size = 14) +
      theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5, color = "#7f8c8d"),
            legend.position = "right")
  })
  
  output$prob_hist <- renderPlot({
    df <- data.frame(
      prob = prob_test, 
      true = factor(ifelse(y_true == 2, "Influential", "Regular"))
    )
    ggplot(df, aes(prob, fill = true)) +
      geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
      geom_vline(xintercept = 0.5, linetype = "dashed", color = "red", size = 1) +
      scale_fill_manual(values = c("Influential" = "#d95f0e", "Regular" = "#2166ac")) +
      labs(title = "Prediction Probability Distribution (Test Set)", 
           x = "P(Influential)", y = "Count", fill = "True Label") +
      theme_minimal(base_size = 14) +
      theme(legend.position = "bottom")
  })
  
  output$metrics <- renderPrint({
    cat("=== Test Set Performance ===\n\n")
    cat(sprintf("  Accuracy:  %.1f%%\n", acc))
    cat(sprintf("  Precision: %.1f%%\n", pre))
    cat(sprintf("  Recall:    %.1f%%\n\n", rec))
    cat(sprintf("Test Set Size: %d nodes\n", length(test_idx)))
    cat(sprintf("Influential nodes: %d (%.1f%%)\n", 
                sum(y_true == 2), 
                sum(y_true == 2)/length(y_true)*100))
    cat(sprintf("Regular nodes: %d (%.1f%%)\n", 
                sum(y_true == 1), 
                sum(y_true == 1)/length(y_true)*100))
  })
}

shinyApp(ui, server)