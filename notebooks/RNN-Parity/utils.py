# ===================================
#  LEARNING PROGRESS HELPER FUNCTION
# ===================================

class LearningProgress: 
    def __init__(self, loop_length=100):
        self.increment_size = 100.0/loop_length
        self.curr_count = 0
        self.curr_pct = 0
        print('LEARNED SEQUENCE LENGTHS: ', end='')
    
    def increment(self):
        self.curr_count += self.increment_size
        if int(self.curr_count) > self.curr_pct:
            self.curr_pct = int(self.curr_count)
            if self.curr_pct <= 100:
                print(self.curr_pct, end=' ')


# ================================
#  PARITY DISPLAY HELPER FUNCTION
# ================================

def echo_even_or_odd(descriptor, num_seq):
    not_div_by_2 = num_seq.sum(axis=(1,)) % 2
    message = 'ODD number of 1s' if not_div_by_2 else 'EVEN number of 1s'
    print('{:<25}{}'.format(descriptor, message))


# ==================================
#  DISPLAY ACCURACY HELPER FUNCTION
# ==================================

def display_accuracy(nn_size, acc_val):
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = [11, 0.75]
    fig, ax = plt.subplots()
    measurements = ('ANN',)
    accuracy = acc_val * 100
    ax.barh(0, accuracy, align='center', color='deepskyblue', ecolor='black')
    ax.set_xticks(range(0, 101, 5))
    ax.set_yticks([0])
    ax.set_yticklabels(measurements)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy')
    ax.set_title('How accurate is our ' + nn_size + ' Neuron Network?')
    plt.show()


class WidgetBuilder(object):
    def __init__(self):
        pass

    def build_learned_seq_widget(self):
        from ipywidgets import FloatProgress, HBox, VBox, HTML, Layout, Button, Box
        learned_seq_desc = Button(description='Learning Sequence', min_width='140px')
        learned_seq_desc.style.button_color = 'lightgreen'
        learned_seq_desc.style.font_weight = 'bold'
        self.learned_seq = HTML()

        label_layout = Layout(
            display='flex',
            flex_flow='row',
            flex='0 0 auto',
            width='150px'
        )

        learned_seq_layout = Layout(display='flex',
                                    flex_flow='row',
                                    align_items='flex-start',
                                    align_content='flex-start',
                                    width='100%')

        learned_seq_desc_widget = Box(children=[learned_seq_desc], layout=label_layout)

        return Box(children=[learned_seq_desc_widget, self.learned_seq], layout=learned_seq_layout)

    def build_stats_widget(self):
        from ipywidgets import FloatProgress, HBox, VBox, HTML, Layout, Button, Box
        loss_text = HTML('Loss', width='140px')
        self.loss_bar = FloatProgress(min=0.0, max=1.0, description='', height = '10px')
        loss_widget = HBox([ loss_text, self.loss_bar ], width='100%')

        acc_text = HTML('Accuracy', width='140px')
        self.acc_bar = FloatProgress(min=0, max=1.0, description='', height = '10px')
        acc_widget = HBox([ acc_text, self.acc_bar ], width='100%')

        box_layout = Layout(display='flex',
                            flex_flow='row',
                            align_items='stretch',
                            justify_content='space-around',
                            border='1px solid #48A7F2',
                            width='100%')

        return Box(children=[acc_widget, loss_widget], layout=box_layout, box_style='info')
