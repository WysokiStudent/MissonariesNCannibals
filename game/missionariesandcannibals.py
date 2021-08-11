#!/usr/bin/env python3
"""
The Missionaries and Canniblas game.
The objective is to get 3M(Missionaries) and 3C(Cannibals) over the river.
Whenever on one side of the river there are more C than M the game is lost.
"""
import os
import inspect
import sys
import pygame

PATH_TO_MODULE = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LEFT_BANK = 0
RIGHT_BANK = 1
MISSIONARIES = 1
CANNIBALS = 0
NO_SPEED = 0

DEFAULT_FONT = pygame.font.get_default_font()
FONT_SIZE = 48
ANTYALIAS = True
MESSAGE_DISPLAY_TIME = 1000

WINDOW_SIZE = (1084, 300)
WINDOW = pygame.display.set_mode(WINDOW_SIZE)
ARENA = WINDOW.get_rect()

BACKGROUND_IMAGE = pygame.image.load(os.path.join(PATH_TO_MODULE, "riverr.png"))
BACKGROUND_COLOR = pygame.Color("Blue")

FERRY_STEP = -5

MAX_FPS = 120
SLOTS = 8

def sanitize(passengers, key, direction):
    """
    Clean input data from invalid an invalid sequence.
    """
    result = []

    if direction < 0:
        direction = LEFT_BANK
    else:
        direction = RIGHT_BANK

    for index, letter in enumerate(key):
        if index == 0:
            result.append(letter)
            continue
        if letter == "c":
            if len(passengers[direction][CANNIBALS]) > 1:
                result.append(letter)
        elif letter == "m":
            if len(passengers[direction][MISSIONARIES]) > 1:
                result.append(letter)

    return ''.join(result)


def flip_verticaly(surface):
    """
    Flip a surface vertically.
    """
    verticaly = True
    horizontaly = False
    return pygame.transform.flip(surface, verticaly, horizontaly)


def redraw(background_image, actors):
    """
    Redraw the whole window.
    """
    WINDOW.fill(BACKGROUND_COLOR)
    WINDOW.blit(background_image, ARENA)
    for actor in actors:
        WINDOW.blit(actor["surf"], actor["rect"])


def get_key(controls):
    """
    Get the key pressed by the user and transform it into a dictionary key.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return "Confirm"
            if event.key in controls:
                key = controls[event.key]
                return key


def final_message(text, color):
    """
    A goodbye message.
    """
    failure_font = pygame.font.Font(DEFAULT_FONT, FONT_SIZE)
    msg = failure_font.render(text, ANTYALIAS, color)
    msg_box = msg.get_rect()
    msg_box.center = ARENA.center
    WINDOW.blit(msg, msg_box)
    pygame.display.flip()
    pygame.time.wait(MESSAGE_DISPLAY_TIME)


class MnC:
    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        self.background_image = BACKGROUND_IMAGE
        self.gamegraph = {
            "bcccmmm-": {
                "c": "ccmmm-bc",
                "cc": "cmmm-bcc",
                "m": "cccmm-bm",
                "mm": "cccm-bmm",
                "cm": "ccmm-bcm"},
            "ccmmm-bc": {
                "c": "bcccmmm-"},
            "cmmm-bcc": {
                "c": "bccmmm-c",
                "cc": "bcccmmm-"},
            "ccmm-bcm": {
                "c": "bcccmm-m",
                "m":  "bccmmm-c",
                "cm": "bcccmmm"},
            "bccmmm-c": {
                "c": "cmmm-bcc",
                "cc": "mmm-bccc",
                "m": "ccmm-bcm",
                "mm": "ccm-bcmm",
                "cm": "cmm-bccm"},
            "mmm-bccc": {
                "c": "bcmmm-cc",
                "cc": "bccmmm-c"},
            "bcmmm-cc": {
                "c": "mmm-bccc",
                "m": "cmm-bccm",
                "mm": "cm-bccmm",
                "cm": "mm-bcccm"},
            "cm-bccmm": {
                "c": "bccm-cmm",
                "cc": "bcccm-mm",
                "m": "bcmm-ccm",
                "mm": "bcmmm-cc",
                "cm": "bccmm-cm"},
            "bccmm-cm": {
                "c": "cmm-bccm",
                "cc": "mm-bcccm",
                "m": "ccm-bcmm",
                "mm": "cc-bcmmm",
                "cm": "cm-bccmm"},
            "cc-bcmmm": {
                "c": "bccc-mmm",
                "m": "bccm-cmm",
                "mm": "bccmm-cm",
                "cm": "bcccm-mm"},
            "bccc-mmm": {
                "c": "cc-bcmmm",
                "cc": "c-bccmmm"},
            "c-bccmmm": {
                    "c": "bcc-cmmm",
                    "cc": "bccc-mmm",
                    "m": "bcm-ccmm",
                    "mm": "bcmm-ccm",
                    "cm": "bccm-cmm"},
            "bcc-cmmm": {
                    "c": "c-bccmmm",
                    "cc": "-bcccmmm"},
            "bcm-ccmm": {
                    "c": "m-bcccmm",
                    "m": "c-bccmmm",
                    "cm": "-bcccmmm"},
            "ccmm-bcm": {
                    "c": "bcccmm-m",
                    "m": "bccmmm-c",
                    "cm": "bcccmmm-"},
            'm-bcccmm': "failure",
            "cccmm-bm": "failure",
            "cccm-bmm": "failure",
            "ccm-bcmm": "failure",
            "bcccmm-m": "failure",
            "cmm-bccm": "failure",
            "mm-bcccm": "failure",
            "bccm-cmm": "failure",
            "bcccm-mm": "failure",
            "bcmm-ccm": "failure",
            "-bcccmmm": "success"
        }

        self.reset()

        self.display_on = True
        self.fpsClock = pygame.time.Clock()
        self.redraw()

    def get_passengers(self, key, direction):
        """
        Returns a list of passengers for transport as well as modifies the
        structure responsible for maintaining a list of who exactly is on which
        side.
        """
        result = []

        if direction < 0:
            direction = LEFT_BANK
            opposite_direction = RIGHT_BANK
        else:
            direction = RIGHT_BANK
            opposite_direction = LEFT_BANK

        for letter in key:
            if letter == "c":
                result.append(self.passengers[direction][CANNIBALS].pop())
                self.passengers[opposite_direction][CANNIBALS].append(result[-1])
            elif letter == "m":
                result.append(self.passengers[direction][MISSIONARIES].pop())
                self.passengers[opposite_direction][MISSIONARIES].append(result[-1])

        return result

    def ferry(self, who, step):
        """
        Transport the actor's appearance from one side of the river to the other.
        Transportation may take many calls to this function.
        """
        if self.display_on == False:
            return True

        done = False
        for actor in who:
            actor["rect"] = actor["rect"].move((step, NO_SPEED))
            if not ARENA.contains(actor["rect"]):
                actor["rect"] = actor["rect"].move((-step, NO_SPEED))
                for actorx in who:
                    actorx["surf"] = flip_verticaly(actorx["surf"])
                done = True
        return done

    def redraw(self):
        if self.display_on:
            redraw(self.background_image, self.actors)

    def step(self, action):
        '''
        Apply action.
        Returns:
        state - game state after taking action
        reward - reward received by taking action (-1 for each step, -500 on
        loss and 500 on victory)
        done - True if a final state is reached, otherwise False
        score - hidden score of the game, essentially a move counter + win/loss

        Returns a reward of -2 on invalid action.
        '''
        # Punish repeating the same state, including illegal actions.
        if self.gamestate in self.visited_states:
            self.visited_states[self.gamestate] += 10
        else:
            self.visited_states[self.gamestate] = 0
        reward = -self.visited_states[self.gamestate]

        # Punish illegal actions.
        if action not in self.get_possible_actions(self.gamestate):
            print("illegal move made",
                    self.gamestate,
                    self.get_possible_actions(self.gamestate),
                    action,
                    action in self.get_possible_actions(self.gamestate))
            reward -= 2
            self.score + reward
            return [self.gamestate, reward, False, self.score]

        # Partial success reward
        reward += 10 * self.gamestate.count('m', self.gamestate.find('-'))
        reward += 10 * self.gamestate.count('c', self.gamestate.find('-'))

        ferry_who = self.get_passengers(
            action,
            self.ferry_step)
        ferry_who.append(self.boat)
        self.ferry_step = -self.ferry_step
        sorted_keys = ''.join(sorted(action))
        self.gamestate = self.gamegraph[self.gamestate][sorted_keys]

        while not self.ferry(ferry_who, self.ferry_step):
            if self.display_on:
                self.redraw()
                pygame.display.flip()
                self.fpsClock.tick(MAX_FPS)

        done = False
        if self.gamegraph[self.gamestate] == "failure":
            reward = -500
            done = True
        elif self.gamegraph[self.gamestate] == "success":
            reward = 500
            done = True
        self.score += reward

        return [self.gamestate, reward, done, self.score]

    def main(self):
        """
        The M&C game
        """
        controls = {
            pygame.K_c: "c",
            pygame.K_m: "m"}
        action = "listen"
        new_key = ""
        ferry_who = []
        self.redraw()
        pygame.display.flip()

        while True:
            if action == "listen":
                key = get_key(controls)
                if key in self.gamegraph[self.gamestate]:
                    if len(new_key) >= 2:
                        new_key = ""
                    new_key = new_key + key
                    new_key = sanitize(self.passengers, new_key, self.ferry_step)
                    print(new_key)
                if key == "Confirm":
                    if len(new_key) == 0:
                        continue
                    
                    sanitized_key = sanitize(self.passengers, new_key, self.ferry_step)
                    print(sanitized_key)
                    _, _, done, _ = self.step(sanitized_key)
                    new_key = ""

                    if done:
                        if self.gamegraph[self.gamestate] == "failure":
                            action = "failure"
                        elif self.gamegraph[self.gamestate] == "success":
                            action = "success"
                        else:
                            action = "listen"
                    self.redraw()

            if action == "failure":
                final_message("Failure!", pygame.Color("Red"))
                sys.exit()
            elif action == "success":
                final_message("Success!", pygame.Color("Green"))
                sys.exit()
        
    def reset(self):
        missionary1 = {"file": os.path.join(PATH_TO_MODULE, "missionaryy.png")}
        missionary2 = {"file": os.path.join(PATH_TO_MODULE, "missionaryy.png")}
        missionary3 = {"file": os.path.join(PATH_TO_MODULE, "missionaryy.png")}
        cannibal1 = {"file": os.path.join(PATH_TO_MODULE, "canniball.png")}
        cannibal2 = {"file": os.path.join(PATH_TO_MODULE, "canniball.png")}
        cannibal3 = {"file": os.path.join(PATH_TO_MODULE, "canniball.png")}
        self.boat = {"file": os.path.join(PATH_TO_MODULE, "boatt.png")}

        self.actors = (
            missionary1, missionary2, missionary3,
            cannibal1, cannibal2, cannibal3,
            self.boat)

        left_bank = [
                [cannibal1, cannibal2, cannibal3],
                [missionary1, missionary2, missionary3]]
        right_bank = [[], []]

        self.passengers = [left_bank, right_bank]
        for i, actor in enumerate(self.actors):
            actor["surf"] = pygame.image.load(actor["file"])
            actor["surf"] = flip_verticaly(actor["surf"])
            actor["rect"] = actor["surf"].get_rect()
            actor["rect"].midleft = (0, (i+1)*ARENA.height/SLOTS)

        self.gamestate = "bcccmmm-"

        self.ferry_step = FERRY_STEP
        self.score = 0
        self.gamestate = "bcccmmm-"
        self.visited_states = {}

        return self.gamestate

    def get_possible_actions(self, state):
        current_bank = state.split('b')[1].split('-')[0]
        possible_actions = []
        if 'c' in current_bank:
            possible_actions.append('c')
        if 'cc' in current_bank:
            possible_actions.append('cc')
        if 'cm' in current_bank:
            possible_actions.append('cm')
        if 'm' in current_bank:
                possible_actions.append('m')
        if 'mm' in current_bank:
                possible_actions.append('mm')

        return possible_actions

    def turn_on_display(self):
        self.display_on = True
        self.redraw()

    def turn_off_display(self):
        self.display_on = False

if __name__ == "__main__":
    game = MnC()
    game.main()
