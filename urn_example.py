import numpy as np

# Set number of balls drawn from urn
num_balls = 50
# Set number of colors
num_colors = 1
# Set number of draws for convergence of expectation
num_draws = 100
# Ensure that all decision rules are evulated on the same sample draws.
seed = 123
# Set gridsize for share grid to be evaluated
share_gridsize = 0.1
# Set grid for different statistical decision functions to be evaluated
lambda_gridsize = 0.01
lambda_grid = np.arange(0, 1 + lambda_gridsize, lambda_gridsize)


def create_share_grid(num_colors, gridsize):
    prob_grid = np.arange(0, 1, gridsize)
    prob_grid[0] += gridsize / 2
    # The last column will be added in the end
    grid = np.array(np.meshgrid(*[prob_grid] * num_colors)).T.reshape(-1, num_colors)
    # Delete points which has probability larger than one
    grid = grid[np.sum(grid, axis=1) < 1]
    # Add last column for white balls
    grid = np.append(grid, (1 - np.sum(grid, axis=1)).reshape(len(grid), 1), axis=1)
    return grid


def get_payoff(guess, theta):
    """Compute payoff based on true theta and guessed theta. This corresponds to
    consequence + u() function."""
    return 1 - np.linalg.norm(guess - theta) ** 2


def guess_with_rule(lam, n, r):
    """This function represents the class of statistical decision functions we
    evaluate in our example."""
    return lam * r / n + (1 - lam) * 0.5


# Create grid of shares
share_grid = create_share_grid(num_colors, share_gridsize)
# Initialize output container for expected performance
expected_performance = np.empty((lambda_grid.shape[0], share_grid.shape[0]))

# Loop over all decision functions
for lambda_id, lambda_val in enumerate(lambda_grid):
    # Loop over all grids
    for grid_id, grid_point in enumerate(share_grid):
        draw_outcomes = np.empty(num_draws)
        # Set seed to ensure consistent evaluation
        np.random.seed(seed)
        for id_draw in range(num_draws):
            # Draw and select number of colored balls
            draw = np.random.multinomial(num_balls, grid_point)
            colored_balls = draw[:num_colors]
            # Get payoff
            draw_outcomes[id_draw] = get_payoff(
                guess_with_rule(lambda_val, num_balls, colored_balls),
                grid_point[:num_colors],
            )
        # Mean over all drawas to get expectation
        expected_performance[lambda_id, grid_id] = np.mean(draw_outcomes)

# Appy maximin decision criteria
opt_lambda_maximin = lambda_grid[np.argmax(np.min(expected_performance, axis=1))]
# Apply minimax regret (best payoff is always 1) => same lambda as maximin
opt_lambda_minimax_reg = lambda_grid[np.argmin(np.max(1-expected_performance, axis=1))]
# Apply subjective bayes with uniform prior
opt_lambda_bayes = lambda_grid[np.argmax(np.mean(expected_performance, axis=1))]

print(opt_lambda_maximin, opt_lambda_minimax_reg, opt_lambda_bayes)
