from flask import Flask, render_template, request
from final_rating import Reviews
import operator
import logging


app = Flask(__name__)

logging.basicConfig(filename="C:/Users/bhavitvyamalik/Desktop/Training/task 1/Flask App/logname.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

@app.route('/')
def home():
    logger.debug("Loading Home page")
    return render_template('base.html')

@app.route('/category',methods = ['POST', 'GET'])
def function():
    if request.method=='POST':
        text = request.form['review']
        logger.debug("Text retrieved")
        A=Reviews()
        logger.debug("Object created")
        logger.debug("Sending to function call")
        #pred_sentences = ['Wore it for my birthday. Perfect fit.']
        result=list(A.getListPrediction([text]))
        logger.debug("Returned from function")
        var=result[0]['probabilities']
        index, value = max(enumerate(var), key=operator.itemgetter(1))
        #print(index,value)
        '''
        if index==1:
            msg="1"
        elif index==2:
            msg="2"
        elif index==3:
            msg="Category: Jackets"
        elif index==4:
            msg="Category: Tops"
        else:
            msg="Category: Trends"
        '''
        logger.debug("Output values decided")
        return render_template("base.html",msg=index,text=text)

        logger.debug("The End")


if __name__=='__main__':
    app.run(debug=True)
