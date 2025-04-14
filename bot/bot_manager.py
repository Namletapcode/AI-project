from bot.heuristic_dodge import HeuristicDodgeBot
# Các kiểu bot khác có thể được import tương tự

def create_bot(game, bot_type, method=None):
    if bot_type == 'heuristic':
        return HeuristicDodgeBot(game, method)
    # Thêm các lựa chọn khác: 'deep_learning', 'neat', v.v.
    else:
        raise ValueError("Bot type không hợp lệ.")