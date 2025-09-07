create database proto;
use proto;

create table register (
    user_id int auto_increment primary key,
    name varchar(100) not null,
    email varchar(150) unique not null,
    password varchar(255) not null,
    confirm_password varchar(255) not null,
    age int,
    gender varchar(10),
    height decimal(5,2),
    weight decimal(5,2)
);

