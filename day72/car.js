class Car {
    constructor(model, maker, color, year) {
        this.model = model;
        this.maker = maker;
        this.color = color;
        this.year = year;
    }

    drive() {
        console.log('drive');
    }

    stop() {
        console.log('stop');
    }
}

class SuperCar extends Car {
    constructor(model, maker, color, year, batteryCapacity) {
        super(model, maker, color, year);
        this.batteryCapacity = batteryCapacity;
    }

    checkBattery() {
        console.log(`${this.model} has a battery capacity of ${this.battery}`);
    }

    drive() {
        console.log(`${this.model} is driving`)
    }
}

let car = new SuperCar("Benz S", "Benz", "red", 2024, 100);
car.drive();
car.stop();