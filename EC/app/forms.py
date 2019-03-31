from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
	username = StringField('Username', validators=[DataRequired()])
	usermail = StringField('Usermail', validators=[DataRequired()])
	submit = SubmitField('Sign In')

class caseForm(FlaskForm):
	usercase = StringField('Usercase', validators=[DataRequired()])
	submit = SubmitField('Submit')

class chooseForm(FlaskForm):
	defence = BooleanField('Do you think you are in a defence?')
	damage = BooleanField('Do you think you really got damage?')
	submit = SubmitField('Submit')

class chooseTort(FlaskForm):
	battery = BooleanField('Battery')
	assult = BooleanField('Assult')
	fault = BooleanField('False imprisonment')
	notT = BooleanField('Not tort')
	submit = SubmitField('Submit')

class identify(FlaskForm):
	c1 = BooleanField('c1')
	c2 = BooleanField('c2')
	c3 = BooleanField('c3')
	submit = SubmitField('Submit')