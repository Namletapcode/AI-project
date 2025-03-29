from game import Game

if __name__ == '__main__':
    game = Game()
    game.run()
    # game.run_in_another_thread()
    print('This line of code is reached')   # debug: never with the old code (single thread)
