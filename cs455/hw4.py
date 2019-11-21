# hw4.py -*- calculating big numbers -*-
#
# Author: Daniel Choo
# Date:   10/20/19


def fact(n):
    if n <= 1:
        return 1
    else:
        return n*fact(n-1)


def main():
    deck = fact(52)
    fiveHand = (fact(52)/((fact(52-5)*fact(5))))

    print("How many ways can you lay out fifty-two cards in a line?:\n" + str(deck))
    print("\nHow many five card hands from a deck of fifty-two?:\n" + str(fiveHand))


main()
